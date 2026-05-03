import os.path as path
from collections import defaultdict
import random
import torch
import copy
from torch.utils.tensorboard import SummaryWriter
import policy
from transformers import BartTokenizer, BartForConditionalGeneration
import torch.nn.functional as F
import os

eps = 1e-4
abs_path = path.abspath (__file__)
model_path = path.dirname (abs_path)
src_path = path.dirname (model_path)
base_path = path.dirname (src_path)
data_processed_path = path.join (base_path, "data_processed")
dot_before_path = path.join (data_processed_path, "dot_before")
dot_after_path = path.join (data_processed_path, "dot_after")
dot_file_path = path.join (dot_before_path, "0.dot")
black_list_path = path.join (base_path, "blacklist.txt")
parameter_path = path.join (model_path, "parameter")
pth_path = path.join (parameter_path, "model.pth")
runs_dir = path.join (base_path, "runs")
train_log_dir = path.join (runs_dir, "cell_exp")

black_list = set ()

def clone_cells (cells) :
    new_cells = []
    for c in cells :
        new_cells.append (Cell (c.id, c.latent.clone ()))
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
    def __init__ (self, n : int, cells : list[Cell], G : list[list[int]]) :
        self.n = n
        self.G = G
        self.cells = cells

    def print (self) :
        print ("cell number = ", self.n, end="")
        for cell in self.cells :
            cell.print ()
        for u in range (0, self.n) :
            for v in self.G[u] :
                print ("(", u, ",", v, end=") ")

    def deg (self, u : int) :
        return len (self.G[u])

def SubtreeSquareSum (tree : ExpectGraph, u : int, vis : dict[int, bool]) :
    num = 0
    vis[u] = True
    ret = torch.sum (tree.cells[u].latent ** 2) + 1
    for v in tree.G[u] :
        if (vis[v]) :
            continue
        (tot, sqs) = SubtreeSquareSum (tree, v, vis)
        num += tot
        ret += sqs
    return num, ret

def markDeadLeaf (pred : ExpectGraph, u : int, vis : dict[int, bool], correct_alive_prob, correct_copy_prob, device) :
    vis[u] = True
    correct_copy_prob[u] = torch.tensor (0.0, device=device)
    if pred.deg (u) == 1 :
        correct_alive_prob[u] = torch.tensor (0.0, device=device)
    for v in pred.G[u] :
        if vis[v] :
            continue
        markDeadLeaf (pred, v, vis, correct_alive_prob, correct_copy_prob, device)

def dfs (pred : ExpectGraph, target : ExpectGraph, u1 : int, u2 : int, vis1 : dict[int, bool], vis2 : dict[int, bool],
         correct_alive_prob, correct_copy_prob, correct_latent_dict, correct_spread_dict,
         latent_node_lst, spread_node_lst, device) :
    #print ("dfs")
    latent_node_lst.append (u1)
    vis1[u1] = True
    vis2[u2] = True
    correct_latent_dict[u1] = target.cells[u2].latent
    correct_alive_prob[u1] = torch.tensor (1.0, device=device)
    limit = min (len (pred.G[u1]), len (target.G[u2]))
    for i in range (0, limit) :
        v1 = pred.G[u1][i]
        v2 = target.G[u2][i]
        if vis1[v1] :
            continue
        if vis2[v2] :
            continue 
        dfs (pred, target, v1, v2, vis1, vis2, 
             correct_alive_prob, correct_copy_prob, 
             correct_latent_dict, correct_spread_dict,
             latent_node_lst, spread_node_lst, device)
    for i in range (limit, len (pred.G[u1])) :
        u = pred.G[u1][i]
        if vis1[u] :
            continue
        markDeadLeaf (pred, u, vis1, correct_alive_prob, correct_copy_prob, device)
    if len (target.G[u2]) > limit :
        correct_alive_prob[u1] = torch.tensor (1.0, device=device)
        correct_copy_prob[u1] = torch.tensor (1.0, device=device)
        correct_spread_dict[u1] = target.cells[target.G[u2][limit]].latent
        spread_node_lst.append (u1)
    else :
        correct_copy_prob[u1] = torch.tensor (0.0, device=device)

class CellGraph :
    def __init__ (self, n : int, cells : list[Cell], G : list[list[int]], degs : list[int]) :
        self.n = n
        self.cells = copy.deepcopy (cells)
        self.degs = copy.deepcopy (degs)
        self.G = copy.deepcopy (G)
        self.spread = []
        self.receive = []
        for i in range (0, self.n) :
            self.spread.append (self.cells[i].latent)

    def degs_update (self) :
        self.degs = [0 for _ in range (self.n)]
        for i in range (self.n) :
            for j in self.G[i] :
                self.degs[i] += 1

    def is_leaf (self, i) :
        return self.degs[i] == 1

    def print (self) :
        print ("cell number = ", self.n, end="")
        for cell in self.cells :
            cell.print ()
        for u in range (0, self.n) :
            for v in self.G[u] :
                print ("(", u, ",", v, end=") ")
        print ("cell living condition : ")

    def print_degs (self) :
        for i in range (self.n) :
            print ("degs[", end="")
            print (i, end="")
            print ("] = ", end="")
            print (self.degs[i])

    def leaf_num (self) :
        leaf_num = 0
        for i in range (self.n) :
            if self.degs[i] == 1 :
                leaf_num += 1
        return leaf_num

    def getReceive (self) :
        self.receive = []
        for i in range (self.n) :
            latents = []
            if len (self.G[i]) == 0 :
                self.receive.append (self.cells[i].latent)
            else :
                for j in self.G[i] :
                    latents.append (self.spread[j])
                self.receive.append (torch.cat (latents, dim=1))

    def train (self, obj_graph : ExpectGraph, update_policy, spread_policy, 
        copy_policy, alive_policy, writer, global_step, optimizer_update,
        optimizer_spread, optimizer_copy, optimizer_alive, device) :
        self.getReceive ()
        self.degs_update ()
        exp_graph = ExpectGraph (copy.deepcopy (self.n), clone_cells (self.cells), copy.deepcopy (self.G))
        #pre_graph = copy.deepcopy (exp_graph)
        alive_prob_dict = {}
        copy_prob_dict = {}
        alive_sample = {}
        copy_sample = {}
        new_latent = []
        new_spread = []
        expect_add_sum = torch.tensor (0.0, device=device).reshape (1, 1, 1)
        expect_sub_sum = torch.tensor (0.0, device=device).reshape (1, 1, 1)
        for i in range (self.n) :
            update_latent = update_policy (self.cells[i].latent, self.receive[i])
            spread_latent = spread_policy (self.cells[i].latent, self.receive[i])
            new_latent.append (update_latent)
            new_spread.append (spread_latent)
            copy_prob = copy_policy (self.cells[i].latent, self.receive[i])
            if (not self.is_leaf (i)) or self.degs[i] == 0 or self.n <= 2 :            # not leaf or root
                #print ("this case")
                exp_graph.cells[i].latent = update_latent
                copy_prob = copy_policy (self.cells[i].latent, self.receive[i]).reshape (1, 1, 1)
                copy_prob_dict[i] = copy_prob
                alive_prob_dict[i] = torch.tensor (1.0, device=device).reshape (1, 1, 1)
                expect_add_sum += copy_prob
            else :                           # leaf
                alive_prob = alive_policy (self.cells[i].latent, self.receive[i]).reshape (1, 1, 1)
                copy_prob = copy_policy (self.cells[i].latent, self.receive[i]).reshape (1, 1, 1)
                copy_prob_dict[i] = copy_prob
                alive_prob_dict[i] = alive_prob           
                expect_sub_sum += torch.tensor (1.0, device=device).reshape (1, 1, 1) - alive_prob
                expect_add_sum += alive_prob * copy_prob
            #print ("alive chance : ", alive_prob_dict[i])
            #print ("copy chance : ", copy_prob_dict[i])
        #print ("Expect graph generate success.")

        pred_graph_vis = defaultdict (bool)
        target_graph_vis = defaultdict (bool)
        correct_copy_prob = {}
        correct_alive_prob = {}
        correct_latent_dict = {}
        correct_spread_dict = {}
        latent_node_lst = []
        spread_node_lst = []
        dfs (
            pred=exp_graph, 
            target=obj_graph, 
            u1=0, u2=0,
            vis1=pred_graph_vis,
            vis2=target_graph_vis,
            correct_alive_prob=correct_alive_prob,
            correct_copy_prob=correct_copy_prob,
            correct_latent_dict=correct_latent_dict,
            correct_spread_dict=correct_spread_dict,
            latent_node_lst=latent_node_lst,
            spread_node_lst=spread_node_lst,
            device=device
        )
        #print ("latent_node_lst : ", latent_node_lst)
        #print ("spread_node_lst : ", spread_node_lst)

        pred_copy_list = []
        target_copy_list = []
        for i in range (self.n) :
            pred_copy = copy_prob_dict[i]
            target_copy = correct_copy_prob[i]
            pred_copy_list.append (pred_copy.reshape (-1))
            target_copy_list.append (target_copy.reshape (-1))
        pred_tensor = torch.cat (pred_copy_list)
        target_tensor = torch.cat (target_copy_list)
        loss_copy = F.binary_cross_entropy (pred_tensor, target_tensor)
        #print ("copy train success.")
        # train copy policy


        pred_alive_list = []
        target_alive_list = []
        for i in range (self.n) :
            if self.degs[i] == 1 :
                #print ("leaf : ", i)
                pred_alive = alive_prob_dict[i].reshape (-1)
                target_alive = correct_alive_prob[i].reshape (-1)
                pred_alive_list.append (pred_alive)
                target_alive_list.append (target_alive)
        pred_tensor = torch.cat (pred_alive_list)
        target_tensor = torch.cat (target_alive_list)
        loss_alive = F.binary_cross_entropy (pred_tensor, target_tensor)
        #print ("alive train success.")
        # train alive policy

        pred_latent_list = []
        target_latent_list = []
        for i in latent_node_lst :
            pred_latent = new_latent[i]
            target_latent = correct_latent_dict[i]
            pred_latent_list.append (pred_latent)
            target_latent_list.append (target_latent)
        pred_tensor = torch.cat (pred_latent_list)
        target_tensor = torch.cat (target_latent_list)
        loss_latent = F.mse_loss (pred_tensor, target_tensor)
        #print ("update train success.")
        # train update policy

        pred_spread_list = []
        target_spread_list = []
        for i in spread_node_lst :
            pred_spread = new_spread[i]
            target_spread = correct_spread_dict[i]
            pred_spread_list.append (pred_spread.reshape (-1))
            target_spread_list.append (target_spread.reshape (-1))
        loss_spread = 0
        if len (pred_spread_list) > 0 :
            pred_tensor = torch.cat (pred_spread_list)
            target_tensor = torch.cat (target_spread_list)
            loss_spread = F.mse_loss (pred_tensor, target_tensor)
        #print ("spread train success.")
        # train spread policy

        pred_n = torch.tensor (0.0 + self.n, device=device).reshape (1, 1, 1) + expect_add_sum - expect_sub_sum
        target_n = torch.tensor (0.0 + obj_graph.n, device=device).reshape (1, 1, 1)
        loss_size = F.l1_loss (pred_n, target_n)
        print ()
        print ("copy loss = ", loss_copy)
        print ("alive loss = ", loss_alive)
        print ("latent loss = ", loss_latent)
        print ("size loss = ", loss_size)
        total_loss = loss_copy + loss_alive + loss_latent + loss_size
        if len (pred_spread_list) > 0 :
            total_loss += loss_spread
            print ("spread loss = ", loss_spread)
        optimizer_copy.zero_grad ()
        optimizer_alive.zero_grad ()
        optimizer_update.zero_grad ()
        if len (pred_spread_list) > 0 :
            optimizer_spread.zero_grad ()

        total_loss.backward ()

        optimizer_copy.step ()
        optimizer_alive.step ()
        optimizer_update.step ()
        if len (pred_spread_list) > 0 :
            optimizer_spread.step ()

        for i in range (self.n) :
            self.cells[i].latent = new_latent[i].detach ()
            self.spread[i] = new_spread[i].detach ()
        average_alive_prob = 0
        average_copy_prob = 0
        for i in range (self.n) :
            average_alive_prob += alive_prob_dict[i]
            average_copy_prob += copy_prob_dict[i]

            if self.degs[i] == 1 :
                alive_sample[i] = int (random.random () < alive_prob_dict[i].item ())
            else :
                alive_sample[i] = 1
            if alive_sample[i] == 0 :
                copy_sample[i] = 0
            else :
                copy_sample[i] = int (random.random () < copy_prob_dict[i].item ())

        new_map = {}
        new_n = 0
        for i in range (self.n) :
            if alive_sample[i] :
                new_map[i] = new_n
                new_n += 1
        new_G = [[] for _ in range (new_n)]
        for i in range (self.n) :
            if alive_sample[i] :
                for j in self.G[i] :
                    if alive_sample[j] :
                        new_G[new_map[i]].append (new_map[j])
        self.cells = [cell for i, cell in enumerate (self.cells) if alive_sample[i]]
        self.spread = [vec for i, vec in enumerate (self.spread) if alive_sample[i]]
        self.n = new_n
        self.G = new_G
        for i in range (self.n) :
            if copy_sample[i] :
                self.cells.append (Cell (new_n, self.spread[i].detach ()))
                self.G.append ([])
                self.G[i].append (new_n)
                self.G[new_n].append (i)
                self.spread.append (self.spread[i])
                new_n += 1
        self.n = new_n

        self.degs_update ()
        #self.print_degs ()
        #print ("leaf_num = ", end="")
        #print (self.leaf_num ())
        print ("current cell number is ", end="")
        print (self.n, end=", target cell number is ")
        print (obj_graph.n)

    def loadFile (self, file_path : str, tokenizer, model, device) :
        self.n = 0
        self.cells = []
        self.G = []
        self.degs = []
        self.spread = []
        print ("loadFile start")
        ast = AST (file_path)
        cells = []
        for node in ast.nodes :
            ast_kind_input = tokenizer (node.ast_kind, return_tensors="pt").to (device)
            with torch.no_grad () :
                ast_kind_latent = model.model.encoder (**ast_kind_input).last_hidden_state
            content_input = tokenizer (node.content, return_tensors="pt").to (device)
            with torch.no_grad () :
                content_latent = model.model.encoder (**content_input).last_hidden_state
            latent0 = ast_kind_latent.mean (dim=1, keepdim=True).to (device)  # (B, 1, d)
            latent1 = content_latent.mean (dim=1, keepdim=True).to (device)    # (B, 1, d)
            latent2 = torch.tanh (torch.tensor (node.ast_id)).unsqueeze (-1).unsqueeze (-1).unsqueeze (-1).to (device) # (1, 1, 1)
            latent3 = torch.tanh (torch.tensor (node.node_id)).unsqueeze (-1).unsqueeze (-1).unsqueeze (-1).to (device) # (1, 1, 1)
            latent = torch.cat ([latent0, latent1, latent2, latent3], dim=2)   # (1, 1, 1538)
            cells.append (Cell (node.node_id, latent))
        self.n = ast.n
        self.cells = cells
        for i in range (0, self.n) :
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
    with open (black_list_path, "r") as f :
        for line in f :
            file = line.strip ()
            if file :
                black_list.add (file)
    writer = SummaryWriter (train_log_dir)
    device = torch.device ("cuda" if torch.cuda.is_available () else "cpu")
    print ("initialize bart.")
    tokenizer = BartTokenizer.from_pretrained ("facebook/bart-base")
    model = BartForConditionalGeneration.from_pretrained ("facebook/bart-base")
    model = model.to (device)
    print ("initialize success.")
    #ast_kind_pooler = attn_pool_project.AttnPoolProject (dim=768)
    #content_pooler = attn_pool_project.AttnPoolProject (dim=768)
    ast = AST (dot_file_path)
    print (type (dot_file_path))
    ast.print_degs ()
    cells = []
    print (type (ast.nodes))
    for node in ast.nodes :
        ast_kind_input = tokenizer (node.ast_kind, return_tensors="pt").to (device)
        with torch.no_grad () :
            ast_kind_latent = model.model.encoder (**ast_kind_input).last_hidden_state
        content_input = tokenizer (node.content, return_tensors="pt").to (device)
        with torch.no_grad () :
            content_latent = model.model.encoder (**content_input).last_hidden_state
        latent0 = ast_kind_latent.mean (dim=1, keepdim=True).to (device)  # (B, 1, d)
        latent1 = content_latent.mean (dim=1, keepdim=True).to (device)    # (B, 1, d)
        latent2 = torch.tensor (node.ast_id).unsqueeze (-1).unsqueeze (-1).unsqueeze (-1).to (device) # (1, 1, 1)
        latent3 = torch.tensor (node.node_id).unsqueeze (-1).unsqueeze (-1).unsqueeze (-1).to (device) # (1, 1, 1)
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

    update_policy = policy.DeepCell (d=1538, num_layer=8).to (device)
    spread_policy = policy.DeepCell (d=1538, num_layer=8).to (device)
    copy_policy = policy.DicideBlock (d=1538, num_layer=8).to (device)
    alive_policy = policy.DicideBlock (d=1538, num_layer=8).to (device)
    learning_rate = 1e-3
    epoch = 10
    round_times = 100
    
    optimizer_update = torch.optim.Adam (update_policy.parameters (), lr=learning_rate)
    optimizer_spread = torch.optim.Adam (spread_policy.parameters (), lr=learning_rate)
    optimizer_copy = torch.optim.Adam (copy_policy.parameters (), lr=learning_rate)
    optimizer_alive = torch.optim.Adam (alive_policy.parameters (), lr=learning_rate)

    if path.exists (pth_path) :
        check_point = torch.load (pth_path, map_location=device)
        update_policy.load_state_dict (check_point["update_policy"])
        spread_policy.load_state_dict (check_point["spread_policy"])
        copy_policy.load_state_dict (check_point["copy_policy"])
        alive_policy.load_state_dict (check_point["dead_policy"])

    # initialize the policy parameters

    train_tot = 0
    file_num = 0
    for i in range (epoch) :
        print ("epoch : ", end="")
        print (epoch)
        for file in sorted (os.listdir (dot_before_path), key=lambda x : int(x.split('.')[0])) :
            if file in black_list :
                continue
            try :
                #print ("GPU MB :", torch.cuda.memory_allocated () / 1024**2)
                print ("file name is ", end="")
                print (file)
                error_path = os.path.join (dot_before_path, file)
                right_path = os.path.join (dot_after_path, file)
                cg.loadFile (error_path, tokenizer=tokenizer, model=model, device=device)
                aid_graph.loadFile (right_path, tokenizer=tokenizer, model=model, device=device)
                right_expect_graph = ExpectGraph (aid_graph.n, aid_graph.cells, aid_graph.G)
                T = 100
                print ("current cell number is ", end="")
                print (cg.n, end=", target cell number is ")
                print (aid_graph.n)
                for j in range (T) :        
                    print ("T = ", j, end=" ")
                    #print ("GPU MB :", torch.cuda.memory_allocated () / 1024**2)
                    cg.train (right_expect_graph, update_policy, spread_policy, 
                        copy_policy, alive_policy, writer, train_tot, optimizer_update,
                        optimizer_spread, optimizer_copy, optimizer_alive, device)
                    if cg.n <= 2 :
                        break
                file_num += 1
            except Exception as e :
                print (f"[CRASH] {file}: {e}")
                with open (black_list_path, "a") as f :
                    f.write (file + "\n")
                black_list.add (file)
                continue
            train_tot += 1
            if train_tot % round_times == 0 :
                torch.save ({
                    "update_policy" : update_policy.state_dict (),
                    "spread_policy" : spread_policy.state_dict (),
                    "copy_policy" : copy_policy.state_dict (),
                    "dead_policy" : alive_policy.state_dict (),
                }, pth_path)

    return 0

if __name__ == '__main__' :
    main ()
