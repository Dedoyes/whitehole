import torch
import torch.nn as nn
import torch.nn.functional as F

class Mixproj (nn.Module) :
    def __init__ (self, in_dim_1, in_dim_2, in_dim_3, out_dim) :
        super ().__init__ ()
        self.latents_map = nn.Linear (in_dim_1, out_dim)
        self.adj_map = nn.Linear (in_dim_2, out_dim)
        self.alive_map = nn.Linear (in_dim_3, out_dim) 
        self.mlp = nn.Linear (3 * out_dim, out_dim)
    def forward (
        self,
        x, # latents (n, d)
        adj_prob, #  (n, n)
        alive_prob  #(n)
    ) :
        x_emb = self.latents_map (x)
        adj_emb = self.adj_map (adj_prob)
        alive_emb = self.alive_map (alive_prob.unsqueeze (-1))
        return self.mlp (torch.cat ([x_emb, adj_emb, alive_emb], dim=-1))

class SparseTransformerLayer(nn.Module):
    def __init__(
        self,
        d_model=1536,
        d_hidden=2048,
        topk=256
    ):
        super().__init__()
        self.topk = topk
        self.scale = d_model ** 0.5
        self.q_proj = Mixproj (
            in_dim_1=1536,
            in_dim_2=1024,
            in_dim_3=1,
            out_dim=d_model
        )
        self.k_proj = Mixproj (
            in_dim_1=1536,
            in_dim_2=1024,
            in_dim_3=1,
            out_dim=d_model
        )
        self.v_proj = Mixproj (
            in_dim_1=1536,
            in_dim_2=1024,
            in_dim_3=1,
            out_dim=d_model
        )
        self.out_proj = nn.Linear(d_model, d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_hidden),
            nn.GELU(),
            nn.Linear(d_hidden, d_model)
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.alpha = nn.Parameter (torch.tensor (1.0))
        self.para1 = nn.Parameter (torch.tensor (1.0))
        self.para2 = nn.Parameter (torch.tensor (1.0))

    def forward(
        self,
        x,           # (n,d)
        adj_prob,    # (n,n)
        alive_prob   # (n)
    ):
        n, d = x.shape
        q = self.q_proj (x, adj_prob, alive_prob)
        k = self.k_proj (x, adj_prob, alive_prob)
        v = self.v_proj (x, adj_prob, alive_prob)
        attn = torch.matmul(
            q,
            k.transpose(0,1)
        ) / self.scale
        topk_val, topk_idx = torch.topk(
            adj_prob,
            k=min(self.topk, n),
            dim=-1
        )
        mask = torch.full(
            (n,n),
            float("-inf"),
            device=x.device
        )
        mask.scatter_(
            1,
            topk_idx,
            0.0
        )
        attn = attn + mask
        attn = F.softmax(attn, dim=-1)
        out = torch.matmul(attn, v)
        out = self.out_proj(out)
        x = self.norm1(x + self.para1 * out)
        ff = self.ffn(x)
        x = self.norm2(x + self.para2 * ff)
        return x


class AliveProbPredict(nn.Module):
    def __init__(
        self,
        d_model=1536,
        d_hidden=2048,
        num_layers=6,
        topk=256
    ) :
        super().__init__()
        self.input_proj = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU()
        )
        self.layers = nn.ModuleList([
            SparseTransformerLayer(
                d_model=d_model,
                d_hidden=d_hidden,
                topk=topk
            )
            for _ in range(num_layers)
        ])
        self.out_proj = nn.Sequential(
            nn.Linear(d_model, d_hidden),
            nn.GELU(),
            nn.Linear(d_hidden, 1)
        )
        self.delta_scale = nn.Parameter (
            torch.tensor (1.0)
        )

    def forward(
        self,
        latents,       # (n,d)
        alive_prob,    # (n)
        adj_prob       # (n,n)
    ) :
        x = latents
        x = self.input_proj(x)
        for layer in self.layers:
            x = layer(x, adj_prob, alive_prob)
        delta = self.out_proj(x).squeeze(-1)
        logit = torch.logit (alive_prob.clamp (1e-6, 1-1e-6))
        pred_logit = logit + self.delta_scale * delta
        pred_alive_prob = torch.sigmoid (pred_logit)
        return pred_alive_prob

class LatentPredict(nn.Module):
    def __init__(
        self,
        d_model=1536,
        d_hidden=2048,
        num_layers=6,
        topk=256
    ) :
        super().__init__()
        self.input_proj = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU()
        )
        self.layers = nn.ModuleList([
            SparseTransformerLayer(
                d_model=d_model,
                d_hidden=d_hidden,
                topk=topk
            )
            for _ in range(num_layers)
        ])
        self.delta_proj = nn.Sequential(
            nn.Linear(d_model, d_hidden),
            nn.GELU(),
            nn.Linear(d_hidden, d_model)
        )
        self.delta_scale = nn.Parameter(
            torch.tensor(1.0)
        )
    def forward(
        self,
        latents,       # (n,d)
        alive_prob,    # (n)
        adj_prob       # (n,n)
    ) :
        x = latents
        x = self.input_proj(x)
        # transformer propagation
        for layer in self.layers:
            x = layer(x, adj_prob, alive_prob)

        # latent delta
        delta = self.delta_proj(x)
        # small-step latent evolution
        new_latents = (
            latents +
            self.delta_scale * delta
        )
        return new_latents

class AdjProbPredict(nn.Module):
    def __init__(
        self,
        d_model=1536,
        d_hidden=2048,
        num_layers=6,
        topk=256
    ):
        super().__init__()
        self.layers = nn.ModuleList([
            SparseTransformerLayer(
                d_model=d_model,
                d_hidden=d_hidden,
                topk=topk
            )
            for _ in range(num_layers)
        ])
        self.edge_q = nn.Linear(d_model, d_model)
        self.edge_k = nn.Linear(d_model, d_model)
        self.edge_bias = nn.Sequential(
            nn.Linear(2, d_hidden),
            nn.GELU(),
            nn.Linear(d_hidden, 1)
        )

    def forward(
        self,
        latents,       # (n,d)
        alive_prob,    # (n)
        adj_prob       # (n,n)
    ) :
        n, d = latents.shape
        x = latents
        # transformer evolution
        for layer in self.layers:
            x = layer(x, adj_prob, alive_prob)
        # edge features
        q = self.edge_q(x)
        k = self.edge_k(x)
        # similarity
        edge_score = torch.matmul(
            q,
            k.transpose(0,1)
        )
        # adjacency probability
        new_adj_prob = torch.sigmoid(
            edge_score
        )
        # remove self-loop
        eye = torch.eye(
            n,
            device=x.device
        )
        new_adj_prob = new_adj_prob * (
            1 - eye
        )
        new_adj_prob = (
            new_adj_prob + new_adj_prob.transpose (0, 1)
        ) * 0.5
        return new_adj_prob
