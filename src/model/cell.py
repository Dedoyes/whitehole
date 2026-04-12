import os.path as path
import torch
import attn_pool_project
import policy
from transformers import BartTokenizer, BartForConditionalGeneration

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

class CellGraph :
    def __init__ (self, n : int, cells : list[Cell], G : list[list[int]]) :
        self.n = n
        self.cells = cells
        self.alive = []
        self.G = G
        self.spread = []
        self.update_policy = policy.DeepCell (d=1538, num_layer=4)
        self.spread_policy = policy.DeepCell (d=1538, num_layer=4)
        self.alive_policy = policy.DicideBlock (d=1538, num_layer=6)
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
    def update (self, T : int) :
        for t in range (T) :

        

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
    cg = CellGraph (len (cells), cells, ast.G)
    for _ in range (cg.n) :
        cg.alive.append (True)
    cg.print ()
    for cell in cg.cells :
        cg.spread.append (cell.latent)
   
    # initialize cell graph and its spread tensor
    


    return 0

if __name__ == '__main__' :
    main ()
