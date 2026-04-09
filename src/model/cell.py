import os.path as path
import torch
from dataclasses import dataclass
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

@dataclass
class Cell : 
    id : int
    latent : torch.Tensor

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
        self.nodes = {}
        self.degs = {}
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
                    self.nodes[self.n] = ASTNode (node_id, ast_kind, ast_id, content)
                    self.G.append ([])
                else :
                    self.G[node_id].append (int (ast_kind))
                    self.G[int (ast_kind)].append (node_id)

    def print (self) :
        print ("n = ", self.n)
        for i in range (1, self.n + 1) :
            self.nodes[i].print ()
            print ("<", end="")
            for v in self.G[i] :
                print (v, end=" ")
            print (">")

def main () :
    tokenizer = BartTokenizer.from_pretrained ("facebook/bart-base")
    model = BartForConditionalGeneration.from_pretrained ("facebook/bart-base")
    ast = AST (dot_file_path)
    ast.print ()
    return 0

if __name__ == '__main__' :
    main ()
