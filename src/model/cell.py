import os.path as path
from collections import defaultdict
import random
import torch
import copy
import policy
from transformers import BartTokenizer, BartForConditionalGeneration
import torch.nn.functional as F
import os

abs_path = path.abspath (__file__)
model_path = path.dirname (abs_path)
src_path = path.dirname (model_path)
base_path = path.dirname (src_path)
data_processed_path = path.join (base_path, "data_processed")
dot_before_path = path.join (data_processed_path, "dot_before")
dot_after_path = path.join (data_processed_path, "dot_after")
dot_file_path = path.join (dot_before_path, "0.dot")

def clone_cells (cells) :
    new_cells = []
    for c in cells :
        new_cells.append (Cell (c.id, c.latent.detach ().clone ()))
    return new_cells

def get_degs (n : int, G : list[list[int]]) :
    degs = []
    for i in range (0, n) :
        degs.append (0)
    for u in range (0, n) :
        for v in G[u] :
            degs[u] += 1
            degs[v] += 1
    return degs

def is_num (s) :
    return s.isdigit ()

class Cell :
    def __init__ (self, id : int, latent : torch.Tensor) :
        self.id = id
        self.latent = latent

    def print (self) :
        print ("id = ", self.id, end = " ")
        print ("latent = ", self.latent)

    def modulus (self) :
        return torch.sqrt (torch.sum (self.latent ** 2))

class ExpectGraph :
    def __init__ (self, n : int, cells : list[Cell], G : list[list[int]], alive : dict[int, bool]) :
        self.n = 0
        self.cells = []
        for i in range (0, n) :
            if alive[i] :
                self.cells.append (cells[i])
                self.n += 1
        self.G = G
    def print (self) :
        print ("cell number = ", self.n, end="")
        for cell in self.cells :
            cell.print ()
        for u in range (0, self.n) :
            for v in self.G[u] :
                print ("(", u, ",", v, end=") ")

def SubtreeSquareSum (tree : ExpectGraph, u : int, vis : dict[int, bool]) :
    vis[u] = True
    ret = torch.sum (tree.cells[u].latent ** 2)
    for v in tree.G[u] :
        if (vis[v]) :
            continue
        ret += SubtreeSquareSum (tree, v, vis)
    return ret

def dfs (pred : ExpectGraph, target : ExpectGraph, u1 : int, u2 : int, vis1 : dict[int, bool], vis2 : dict[int, bool]) :
    #print ("dfs")
    vis1[u1] = True
    vis2[u2] = True
    latent_pred = pred.cells[u1].latent
    latent_target = target.cells[u2].latent
    ret = F.mse_loss (latent_pred, latent_target)
    limit = min (len (pred.G[u1]), len (target.G[u2]))
    for i in range (0, limit) :
        v1 = pred.G[u1][i]
        v2 = target.G[u2][i]
        if (vis1[v1]) :
            continue
        if (vis2[v2]) :
            continue
        ret += dfs (pred, target, v1,v2, vis1, vis2)
    for i in range (limit, len (pred.G[u1])) :
        u = pred.G[u1][i]
        if (vis1[u]) :
            continue
        ret += SubtreeSquareSum (pred, u, vis1)
    for i in range (limit, len (target.G[u2])) :
        u = target.G[u2][i]
        if (vis2[u]) :
            continue
        ret += SubtreeSquareSum (target, u, vis2)
    return ret

def ExpectGraphMSE (g1 : ExpectGraph, g2 : ExpectGraph) :
    vis1 = defaultdict (bool)
    vis2 = defaultdict (bool)
    return dfs (g1, g2, 0, 0, vis1, vis2)

class CellGraph :
    def __init__ (self, n : int, cells : list[Cell], G : list[list[int]], degs : list[int]) :
        self.n = n
        self.cells = copy.deepcopy (cells)
        self.alive = {}
        for i in range (0, self.n) :
            self.alive[i] = True
        self.degs = copy.deepcopy (degs)
        self.G = copy.deepcopy (G)
        self.spread = []
        self.receive = []
        for i in range (0, self.n) :
            self.spread.append (self.cells[i].latent)

    def print (self) :
        print ("cell number = ", self.n, end="")
        for cell in self.cells :
            cell.print ()
        for u in range (0, self.n) :
            for v in self.G[u] :
                print ("(", u, ",", v, end=") ")
        print ("cell living condition : ")
        for x in self.alive :
            print (x, end=" ")

    def getReceive (self) :
        self.receive = []
        for i in range (self.n) :
            if self.alive[i] :
                latents = []
                for j in self.G[i] :
                    #print ("i = ", end="")
                    #print (i, end=" ")
                    #print ("j = ", end="")
                    #print (j)
                    if self.alive[j] :
                        latents.append (self.spread[j])
                self.receive.append (torch.cat (latents, dim=1).detach ())
            else :
                self.receive.append (None)

    def train (self, obj_graph : ExpectGraph, update_policy, spread_policy, 
        copy_policy, dead_policy, learning_rate=1e-3) :
        print ("current cell number is ", end="")
        print (self.n)
        optimizer = torch.optim.Adam (
            list (update_policy.parameters ()) +
            list (spread_policy.parameters ()) +
            list (copy_policy.parameters ()) +
            list (dead_policy.parameters ()),
            lr = learning_rate
        )
        self.getReceive ()
        exp_graph = ExpectGraph (copy.deepcopy (self.n), clone_cells (self.cells), copy.deepcopy (self.G), copy.deepcopy (self.alive))
        #pre_graph = copy.deepcopy (exp_graph)
        alive_prob_dict = {}
        copy_prob_dict = {}
        alive_sample = {}
        copy_sample = {}
        new_latent = []
        new_spread = []
        for i in range (self.n) :
            if self.alive[i] :
                update_latent = update_policy (self.cells[i].latent, self.receive[i])
                spread_latent = spread_policy (self.cells[i].latent, self.receive[i])
                new_latent.append (update_latent)
                new_spread.append (spread_latent)
                self.spread[i] = spread_latent.detach ()
                copy_prob = copy_policy (self.cells[i].latent, self.receive[i])
                if self.degs[i] != 1 :
                    exp_graph.cells[i].latent = update_latent
                    copy_prob = copy_policy (self.cells[i].latent, self.receive[i])
                    exp_graph.cells.append (Cell (exp_graph.n, (spread_latent * copy_prob).detach ()))
                    exp_graph.G.append ([])
                    exp_graph.G[exp_graph.n].append (i)
                    exp_graph.G[i].append (exp_graph.n)
                    exp_graph.n += 1 
                    copy_prob_dict[i] = copy_prob
                    alive_prob_dict[i] = torch.tensor (1.0)
                else :
                    dead_prob = dead_policy (self.cells[i].latent, self.receive[i])
                    copy_prob = copy_policy (self.cells[i].latent, self.receive[i])
                    exp_graph.cells[i].latent = ((1.0 - dead_prob) * exp_graph.cells[i].latent).detach ()
                    exp_graph.cells.append (Cell (exp_graph.n, (spread_latent * copy_prob * (1.0 - dead_prob).detach ())))
                    exp_graph.G.append ([])
                    exp_graph.G[exp_graph.n].append (i)
                    exp_graph.G[i].append (exp_graph.n)
                    exp_graph.n += 1
                    copy_prob_dict[i] = copy_prob
                    alive_prob_dict[i] = torch.tensor (1.0) - dead_prob
            else :
                new_latent.append (None)
                new_spread.append (None)
        mse_loss = ExpectGraphMSE (exp_graph, obj_graph)
        optimizer.zero_grad ()
        mse_loss.backward ()
        optimizer.step ()
        print ("train complete")
        # train the policy network
        print ("mse_loss = ", end="")
        print (mse_loss)
        for i in range (self.n) :
            if self.alive[i] :
                alive_sample[i] = int (random.random () < alive_prob_dict[i])
                if alive_sample[i] == 0 :
                    copy_sample[i] = 0
                else :
                    copy_sample[i] = int (random.random () < copy_prob_dict[i])
        # caculate the alive and copy probility
        pre_n = self.n
        for i in range (pre_n) :
            if self.alive[i] :
                if alive_sample[i] :
                    self.cells[i].latent = new_latent[i].detach ()
                    if copy_sample[i] :
                        new_id = self.n
                        self.n += 1
                        print ("when i is ", i, end=" ,")
                        print ("new_id is", new_id, end="")
                        self.G.append ([])
                        self.G[i].append (new_id)
                        self.G[new_id].append (i)
                        self.cells.append (Cell (new_id, new_spread[i].detach ()))
                        self.spread.append (new_spread[i].detach ())
                        self.alive[new_id] = True
                        print ("new_id = ", end="")
                        print (new_id)
                        self.degs.append (1)
                        self.degs[i] += 1
                else :
                    for v in self.G[i] :
                        if self.alive[v] :
                            self.degs[v] -= 1
                    self.alive[i] = False

    def loadFile (self, file_path : str, tokenizer, model) :
        print ("loadFile start")
        ast = AST (file_path)
        cells = []
        for node in ast.nodes :
            ast_kind_input = tokenizer (node.ast_kind, return_tensors="pt")
            with torch.no_grad () :
                ast_kind_latent = model.model.encoder (**ast_kind_input).last_hidden_state
            content_input = tokenizer (node.content, return_tensors="pt")
            with torch.no_grad () :
                content_latent = model.model.encoder (**content_input).last_hidden_state
            latent0 = ast_kind_latent.mean (dim=1, keepdim=True)  # (B, 1, d)
            latent1 = content_latent.mean (dim=1, keepdim=True)    # (B, 1, d)
            latent2 = torch.tensor (node.ast_id).unsqueeze (-1).unsqueeze (-1).unsqueeze (-1) # (1, 1, 1)
            latent3 = torch.tensor (node.node_id).unsqueeze (-1).unsqueeze (-1).unsqueeze (-1) # (1, 1, 1)
            latent = torch.cat ([latent0, latent1, latent2, latent3], dim=2)   # (1, 1, 1538)
            cells.append (Cell (node.node_id, latent))
        self.n = ast.n
        self.cells = cells
        self.alive = {}
        for i in range (0, self.n) :
            self.alive[i] = True
            self.spread.append (self.cells[i].latent)
        self.degs = get_degs (self.n, ast.G)
        self.G = ast.G
        print ("loadFile end")


class ASTNode :
    def __init__ (self, node_id : int, ast_kind : str, ast_id : int, content : str) :
        self.node_id = node_id
        self.ast_kind = ast_kind
        self.ast_id = ast_id
        self.content = content
    
    def print (self) :
        print (self.node_id, end=" ")
        print (self.ast_kind, end=" ")
        print (self.ast_id, end=" ")
        print (self.content)

class AST :
    def __init__ (self, dot_path : str) :
        self.n = 0
        self.nodes = []
        self.degs = []
        self.G = []
        self.G.append ([])
        with open (dot_path, "r") as f :
            for line in f :
                parts = line.strip ().split (maxsplit=3)
                if not parts :
                    continue
                node_id = int (parts[0])
                ast_kind = parts[1]
                if not is_num (ast_kind) :
                    ast_id = int (parts[2])
                    if len (parts) == 4 : 
                        content = parts[3]
                    else :
                        content = "None"
                    self.n += 1
                    self.nodes.append (ASTNode (node_id - 1, ast_kind, ast_id, content))
                    self.degs.append (0)
                    self.G.append ([])
                else :
                    self.G[node_id - 1].append (int (ast_kind) - 1)
                    self.G[int (ast_kind) - 1].append (node_id - 1)
                    self.degs[node_id - 1] += 1
                    self.degs[int (ast_kind) - 1] += 1

    def print (self) :
        print ("n = ", self.n)
        for i in range (0, self.n) :
            self.nodes[i].print ()
            print ("<", end="")
            for v in self.G[i] :
                print (v, end=" ")
            print (">")
    
    def print_degs (self) :
        print ("[", end="")
        for x in self.degs :
            print (x, end=" ")
        print ("]")
        

def main () :
    print ("initialize bart.")
    tokenizer = BartTokenizer.from_pretrained ("facebook/bart-base")
    model = BartForConditionalGeneration.from_pretrained ("facebook/bart-base")
    print ("initialize success.")
    #ast_kind_pooler = attn_pool_project.AttnPoolProject (dim=768)
    #content_pooler = attn_pool_project.AttnPoolProject (dim=768)
    ast = AST (dot_file_path)
    print (type (dot_file_path))
    ast.print_degs ()
    cells = []
    print (type (ast.nodes))
    for node in ast.nodes :
        ast_kind_input = tokenizer (node.ast_kind, return_tensors="pt")
        with torch.no_grad () :
            ast_kind_latent = model.model.encoder (**ast_kind_input).last_hidden_state
        content_input = tokenizer (node.content, return_tensors="pt")
        with torch.no_grad () :
            content_latent = model.model.encoder (**content_input).last_hidden_state
        latent0 = ast_kind_latent.mean (dim=1, keepdim=True)  # (B, 1, d)
        latent1 = content_latent.mean (dim=1, keepdim=True)    # (B, 1, d)
        latent2 = torch.tensor (node.ast_id).unsqueeze (-1).unsqueeze (-1).unsqueeze (-1) # (1, 1, 1)
        latent3 = torch.tensor (node.node_id).unsqueeze (-1).unsqueeze (-1).unsqueeze (-1) # (1, 1, 1)
        latent = torch.cat ([latent0, latent1, latent2, latent3], dim=2)   # (1, 1, 1538)
        cells.append (Cell (node.node_id, latent))
    print (len (cells))
    for cell in cells :
        cell.print ()
        #break
    cg = CellGraph (len (cells), cells, ast.G, ast.degs)
    aid_graph = copy.deepcopy (cg)
    cg.print ()
    # initialize cell graph and its spread tensor
    
    update_policy = policy.DeepCell (d=1538, num_layer=4)
    spread_policy = policy.DeepCell (d=1538, num_layer=4)
    copy_policy = policy.DicideBlock (d=1538, num_layer=6)
    dead_policy = policy.DicideBlock (d=1538, num_layer=6)
    learning_rate = 1e-3
    epoch = 10
    T = 3
    # initialize the policy parameters

    file_num = 0
    for i in range (epoch) :
        print ("epoch : ", end="")
        print (epoch)
        for file in os.listdir (dot_before_path) :
            print ("file name is ", end="")
            print (file)
            error_path = os.path.join (dot_before_path, file)
            right_path = os.path.join (dot_after_path, file)
            cg.loadFile (error_path, tokenizer=tokenizer, model=model)
            aid_graph.loadFile (right_path, tokenizer=tokenizer, model=model)
            right_expect_graph = ExpectGraph (aid_graph.n, aid_graph.cells, aid_graph.G, aid_graph.alive)
            for t in range (T) :
                cg.train (right_expect_graph, update_policy, spread_policy, copy_policy, dead_policy,
                          learning_rate=learning_rate)
            file_num += 1
    
    return 0

if __name__ == '__main__' :
    main ()
