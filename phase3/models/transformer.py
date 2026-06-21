"""
ObserverAggregatorTransformer — identical architecture to crypto/phase3,
modality dims updated for crypto supplementary sources.
"""

import math
from typing import Dict, Tuple
import torch, torch.nn as nn, torch.nn.functional as F
from phase3.utils.config import D_MODEL, NUM_HEADS, NUM_LAYERS, D_FF, DROPOUT, MODALITY_DIMS, SIGMA_DIM


class _MHA(nn.Module):
    def __init__(self, d, h, drop):
        super().__init__()
        self.dk = d // h; self.h = h
        self.Wq = nn.Linear(d,d); self.Wk = nn.Linear(d,d)
        self.Wv = nn.Linear(d,d); self.Wo = nn.Linear(d,d)
        self.drop = nn.Dropout(drop)
    def forward(self, q, k, v):
        B = q.size(0)
        Q = self.Wq(q).view(B,-1,self.h,self.dk).transpose(1,2)
        K = self.Wk(k).view(B,-1,self.h,self.dk).transpose(1,2)
        V = self.Wv(v).view(B,-1,self.h,self.dk).transpose(1,2)
        s = torch.matmul(Q,K.transpose(-2,-1)) / math.sqrt(self.dk)
        w = self.drop(F.softmax(s,dim=-1))
        o = torch.matmul(w,V).transpose(1,2).contiguous().view(B,-1,self.h*self.dk)
        return self.Wo(o)


class _EncLayer(nn.Module):
    def __init__(self, d, h, ff, drop):
        super().__init__()
        self.attn  = _MHA(d,h,drop)
        self.ff    = nn.Sequential(nn.Linear(d,ff),nn.GELU(),nn.Dropout(drop),nn.Linear(ff,d))
        self.n1    = nn.LayerNorm(d); self.n2 = nn.LayerNorm(d)
        self.drop  = nn.Dropout(drop)
    def forward(self, x):
        x = self.n1(x + self.drop(self.attn(x,x,x)))
        x = self.n2(x + self.drop(self.ff(x)))
        return x


class ObserverAggregatorTransformer(nn.Module):
    def __init__(self, d_model=D_MODEL, num_heads=NUM_HEADS,
                 num_layers=NUM_LAYERS, d_ff=D_FF, dropout=DROPOUT):
        super().__init__()
        self.num_mod = len(MODALITY_DIMS)
        self.mod_proj = nn.ModuleDict(
            {name: nn.Linear(dim, d_model) for name, dim in MODALITY_DIMS.items()})
        self.mod_emb  = nn.Parameter(torch.randn(self.num_mod, d_model) * 0.02)
        self.cls      = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
        self.encoder  = nn.ModuleList([_EncLayer(d_model,num_heads,d_ff,dropout)
                                       for _ in range(num_layers)])
        self.proj = nn.Sequential(nn.Linear(d_model,SIGMA_DIM),nn.GELU(),
                                  nn.Dropout(dropout),nn.Linear(SIGMA_DIM,SIGMA_DIM))
        self._init()

    def _init(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None: nn.init.zeros_(m.bias)

    def _split(self, x):
        out, i = {}, 0
        for name, dim in MODALITY_DIMS.items():
            out[name] = x[:, i:i+dim]; i += dim
        return out

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        B = x.size(0)
        mods = self._split(x)
        tokens = torch.stack([self.mod_proj[n](mods[n]) for n in MODALITY_DIMS], dim=1)
        tokens = tokens + self.mod_emb.unsqueeze(0)
        tokens = torch.cat([self.cls.expand(B,-1,-1), tokens], dim=1)
        for layer in self.encoder:
            tokens = layer(tokens)
        cls_out  = tokens[:, 0, :]
        mod_out  = tokens[:, 1:, :]
        sigma_ai = self.proj(cls_out)

        # Pairwise cosine alignment
        normed = F.normalize(mod_out, p=2, dim=2)
        sim    = torch.bmm(normed, normed.transpose(1,2))
        n = self.num_mod
        mask = torch.triu(torch.ones(n,n,device=sim.device),diagonal=1).bool()
        alpha = sim[:, mask].mean(dim=1, keepdim=True)
        return sigma_ai, alpha

    def param_count(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
