import os.path as path
import torch
import attn_pool_project
import policy
from transformers import BartTokenizer, BartForConditionalGeneration
import torch.nn.functional as F

abs_path = path.abspath (__file__)
model_path = path.dirname (abs_path)
src_path = path.dirname (model_path)
base_path = path.dirname (src_path)
data_processed_path = path.join (base_path, "data_processed")
dot_before_path = path.join (data_processed_path, "dot_before")
dot_file_path = path.join (dot_before_path, "0.dot")

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
        if (vis2[v1]) :
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
    vis1 = {}
    vis2 = {}
    return dfs (g1, g2, 0, 0, vis1, vis2)

class CellGraph :
    def __init__ (self, n : int, cells : list[Cell], G : list[list[int]], degs : list[int]) :
        self.n = n
        self.cells = cells
        self.alive = {}
        for i in range (0, self.n) :
            self.alive[i] = True
        self.degs = degs
        self.G = G
        self.spread = []
        self.receive = []
        self.update_policy = policy.DeepCell (d=1538, num_layer=4)
        self.spread_policy = policy.DeepCell (d=1538, num_layer=4)
        self.copy_policy = policy.DicideBlock (d=1538, num_layer=6)
        self.dead_policy = policy.DicideBlock (d=1538, num_layer=6)
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
    def update (self, obj_graph : ExpectGraph) :
        for i in range (self.n) :
            if self.alive[i] :
                latents = []
                for j in self.G[i] :
                    if self.alive[j] :
                        latents.append (self.spread[j])
                self.receive[i] = torch.cat (latents, dim=1)
        pre_graph = ExpectGraph (self.n, self.cells, self.G, self.alive)
        exp_graph = ExpectGraph (self.n, self.cells, self.G, self.alive)
        for i in range (self.n) :
            if self.alive[i] :
                update_latent = self.update_policy (self.cells[i].latent, self.receive[i])
                spread_latent = self.spread_policy (self.cells[i].latent, self.receive[i])
                copy_prob = self.copy_policy (self.cells[i].latent, self.receive[i])
                if self.degs != 1 :
                    exp_graph.cells[i].latent = update_latent
                    copy_prob = self.copy_policy (self.cells[i].latent, self.receive[i])
                    exp_graph.cells.append (Cell (exp_graph.n, spread_latent * copy_prob))
                    exp_graph.G[exp_graph.n].append (i)
                    exp_graph.G[i].append (exp_graph.n)
                    exp_graph.n += 1 
                else :
                    dead_prob = self.dead_policy (self.cells[i].latent, self.receive[i])
                    copy_prob = self.copy_policy (self.cells[i].latent, self.receive[i])
                    exp_graph.cells[i].latent = (1.0 - dead_prob) * exp_graph.cells[i].latents
                    exp_graph.cells.append (Cell (exp_graph.n, spread_latent * copy_prob * (1.0 - dead_prob)))
                    exp_graph.G[exp_graph.n].append (i)
                    exp_graph.G[i].append (exp_graph.n)
                    exp_graph.n += 1
        pre_dis = ExpectGraphMSE (pre_graph, obj_graph)
        exp_dis = ExpectGraphMSE (exp_graph, obj_graph)

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
                    content = parts[3]
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
    ast_kind_pooler = attn_pool_project.AttnPoolProject (dim=768)
    content_pooler = attn_pool_project.AttnPoolProject (dim=768)
    ast = AST (dot_file_path)
    ast.print_degs ()
    cells = []
    print (type (ast.nodes))
    for node in ast.nodes :
        ast_kind_input = tokenizer (node.ast_kind, return_tensors="pt")
        ast_kind_latent = model.model.encoder (**ast_kind_input).last_hidden_state
        content_input = tokenizer (node.content, return_tensors="pt")
        content_latent = model.model.encoder (**content_input).last_hidden_state
        latent0 = ast_kind_pooler (ast_kind_latent).unsqueeze (1)  # (B, 1, d)
        latent1 = content_pooler (content_latent).unsqueeze (1)    # (B, 1, d)
        latent2 = torch.tensor (node.ast_id).unsqueeze (-1).unsqueeze (-1).unsqueeze (-1) # (1, 1, 1)
        latent3 = torch.tensor (node.node_id).unsqueeze (-1).unsqueeze (-1).unsqueeze (-1) # (1, 1, 1)
        latent = torch.cat ([latent0, latent1, latent2, latent3], dim=2)   # (1, 1, 1538)
        cells.append (Cell (node.node_id, latent))
    print (len (cells))
    for cell in cells :
        cell.print ()
        #break
    cg = CellGraph (len (cells), cells, ast.G, ast.degs)
    for _ in range (cg.n) :
        cg.alive.append (True)
    cg.print ()
    for cell in cg.cells :
        cg.spread.append (cell.latent)
   
    # initialize cell graph and its spread tensor
    


    return 0

if __name__ == '__main__' :
    main ()
