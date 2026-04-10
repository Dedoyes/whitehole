import torch.nn as nn
import torch

class AttnPoolProject (nn.Module) :
    def __init__ (self, dim=768) :
        super ().__init__ ()
        self.query = nn.Parameter (torch.randn (dim))
        self.proj = nn.Linear (dim, dim)

    def forward (self, x) :   # x : (B, n, d)
        scores = x @ self.query  # (B, n)
        weights = torch.softmax (scores, dim=1)
        pooled = weights.unsqueeze (-1).permute (0, 2, 1) @ x   # (B, 1, d)
        return self.proj (pooled.squeeze (1))  # (B, d)
        
