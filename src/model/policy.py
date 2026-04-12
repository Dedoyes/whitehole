import torch
import torch.nn as nn
import math

class CellUpdate (nn.Module) :
    def __init__ (self, d=1538) :
        super ().__init__ ()
        self.d = d
        self.q = nn.Linear (d, d)
        self.k = nn.Linear (d, d)
        self.v = nn.Linear (d, d)
        self.mlp = nn.Sequential (
            nn.Linear (2 * d, d),
            nn.ReLU (),
            nn.Linear (d, d)
        )
        self.norm1 = nn.LayerNorm (d)
        self.norm2 = nn.LayerNorm (d)
        self.alpha = nn.Parameter (torch.tensor (0.1))

    def forward (self, self_latent : torch.Tensor, neighbors_latent : torch.Tensor) : # x : (B, n, d)
        q = self.q (self_latent)
        k = self.k (neighbors_latent)
        v = self.v (neighbors_latent)
        attn = torch.softmax (
            q @ k.transpose (-2, -1) / math.sqrt (self.d),
            dim=-1
        )   # (B, 1, n)
        agg = attn @ v # (B, 1, d)
        h = self_latent + agg
        h = self.norm1 (h)
        delta = self.mlp (torch.cat ([self_latent, h], dim=-1))
        out = self_latent + torch.tanh (self.alpha) * delta
        out = self.norm2 (out)
        return out   # (B, 1, d)

class DeepCell (nn.Module) :
    def __init__ (self, d=1538, num_layer=4) :
        super ().__init__ ()
        self.layers = nn.ModuleList ([
            CellUpdate (d) for _ in range (num_layer)
        ])

    def forward (self, self_latent : torch.Tensor, neighbors_latent : torch.Tensor) : # (B, n, d)
        h = self_latent.clone ()
        for layer in self.layers :
            h = layer (h, neighbors_latent)
        return h   # (B, 1, d)

class DicideBlock (nn.Module) :
    def __init__ (self, d=1538, num_layer=6) :
        super ().__init__ ()
        self.layer = DeepCell (d, num_layer)
        self.linear = nn.Linear (d, 1)
    def forward (self, self_latent : torch.Tensor, neighbors_latent : torch.Tensor) :
        x = self_latent.clone ()
        x = self.layer (x, neighbors_latent)
        x = self.linear (x)
        x = torch.sigmoid (x)
        return x
     
        
