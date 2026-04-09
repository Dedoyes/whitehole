import os.path as path
import torch
from dataclasses import dataclass

abs_path = path.abspath (__file__)
model_path = path.dirname (abs_path)
src_path = path.dirname (model_path)
base_path = path.dirname (src_path)
data_processed_path = path.join (base_path, "data_processed")
dot_before_path = path.join (data_processed_path, "dot_before")

@dataclass
class Cell : 
    id : int
    latent : torch.Tensor

class Latent :
    latent : torch.Tensor

class CellGraph :
    def __init__ (self, dot_path) :
        self.cells = {}
        self.spread = {}
        self.G = []

    
