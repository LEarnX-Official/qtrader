"""
Phase 5 Actor-Critic.

State  : 8-token observation (4530 dims) — BTC included as context
Action : 7-token portfolio weights       — BTC excluded from trading

The per-asset encoder processes all 8 token embeddings through cross-asset
attention, then a separate 7-token head produces the tradeable weights.
BTC's embedding influences the others through attention but gets no weight.
"""

from typing import List, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

from phase5.utils.config import (
    ASSET_EMBED_DIM, DROPOUT, HIDDEN_DIM,
    N_ASSETS, N_STATE_TOKENS, NUM_HEADS,
    PER_ASSET_DIM, STATE_DIM,
)


class _AssetAttn(nn.Module):
    def __init__(self, dim=ASSET_EMBED_DIM, heads=NUM_HEADS):
        super().__init__()
        assert dim % heads == 0
        self.h = heads; self.hd = dim // heads
        self.q = nn.Linear(dim, dim); self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim); self.o = nn.Linear(dim, dim)

    def forward(self, x):   # (B, N, dim)
        B, N, D = x.shape; H, Hd = self.h, self.hd
        Q = self.q(x).view(B, N, H, Hd).transpose(1, 2)
        K = self.k(x).view(B, N, H, Hd).transpose(1, 2)
        V = self.v(x).view(B, N, H, Hd).transpose(1, 2)
        a = F.softmax((Q @ K.transpose(-2, -1)) / Hd**0.5, dim=-1)
        return self.o((a @ V).transpose(1, 2).contiguous().view(B, N, D))


class ActorNetwork(nn.Module):
    """
    Hierarchical actor:
      1. Encode all 8 token embeddings (including BTC)
      2. Cross-asset attention — BTC influences alts through attention
      3. Extract only the 7 trade-token embeddings for the output heads
      4. Direction + confidence → Kelly-weighted portfolio weights (7 assets)
    """

    def __init__(
        self,
        n_state_tokens: int = N_STATE_TOKENS,   # 8 (all tokens in state)
        n_trade_assets: int = N_ASSETS,          # 7 (tradeable tokens)
        per_asset_dim:  int = PER_ASSET_DIM,
        embed_dim:      int = ASSET_EMBED_DIM,
        hidden_dim:     int = HIDDEN_DIM,
        dropout:        float = DROPOUT,
    ):
        super().__init__()
        self.n_state   = n_state_tokens   # 8
        self.n_trade   = n_trade_assets   # 7
        self.pad       = per_asset_dim    # 565
        ctx_dim        = n_trade_assets + 3   # portfolio weights + pnl/vol/dd

        # Encoder for all 8 token embeddings
        self.enc = nn.Sequential(
            nn.Linear(per_asset_dim, 256), nn.LayerNorm(256), nn.GELU(),
            nn.Dropout(dropout), nn.Linear(256, embed_dim), nn.LayerNorm(embed_dim),
        )
        # Cross-asset attention — BTC at index 0 attends to all alts and vice versa
        self.attn  = _AssetAttn(embed_dim, NUM_HEADS)
        self.anorm = nn.LayerNorm(embed_dim)

        # Portfolio context encoder
        self.ctx_enc = nn.Sequential(
            nn.Linear(ctx_dim, 64), nn.GELU(), nn.Linear(64, embed_dim),
        )

        # Fusion and output heads operate on 7 TRADE tokens only (skip BTC)
        self.fusion = nn.Sequential(
            nn.Linear(embed_dim * 2, hidden_dim), nn.LayerNorm(hidden_dim),
            nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2), nn.GELU(),
        )
        fd = hidden_dim // 2
        # Output dim = n_trade_assets + 1 (7 asset logits + 1 cash logit)
        self.dir_h  = nn.Linear(fd, n_trade_assets + 1)
        self.conf_h = nn.Linear(fd, n_trade_assets + 1)

    def forward(
        self,
        state:  torch.Tensor,            # (B, STATE_DIM=4530)
        kelly:  Optional[torch.Tensor],  # (B, 7) or None
    ) -> torch.Tensor:                   # (B, 8) weights summing to 1 (last = cash)
        B = state.size(0)

        # Split state: 8×565 features + portfolio context (7 weights + 3 metrics)
        per_flat = state[:, :self.n_state * self.pad]        # (B, 8×565=4520)
        ctx_flat = state[:, self.n_state * self.pad:]        # (B, 10)

        # Encode all 8 token embeddings
        x = per_flat.view(B, self.n_state, self.pad)         # (B, 8, 565)
        x = self.enc(x)                                       # (B, 8, embed_dim)

        # Cross-asset attention — BTC (index 0) influences alts through attention
        x = self.anorm(x + self.attn(x))                     # (B, 8, embed_dim)

        # Keep only 7 TRADE token embeddings (indices 1-7, skip BTC at 0)
        x_trade = x[:, 1:, :]                                # (B, 7, embed_dim)

        # Broadcast portfolio context to each trade token
        ctx = self.ctx_enc(ctx_flat).unsqueeze(1).expand(-1, self.n_trade, -1)

        # Fuse and pool
        fused  = self.fusion(torch.cat([x_trade, ctx], dim=-1))  # (B, 7, hd/2)
        pooled = fused.mean(dim=1)                                # (B, hd/2)

        # (B, 8): 7 asset logits + 1 cash logit
        direction  = (torch.tanh(self.dir_h(pooled)) + 1.0) / 2.0   # (B, 8) in [0,1]
        confidence = torch.sigmoid(self.conf_h(pooled))               # (B, 8) in [0,1]

        # Asset weights use Kelly; cash score = 1 - mean(kelly) so the agent
        # is automatically nudged toward cash when all assets look unattractive.
        if kelly is not None:
            # kelly: (B, 7) — higher mean Kelly → less cash, lower → more cash
            cash_score = (1.0 - kelly.mean(dim=1, keepdim=True)).clamp(0.0, 1.0)  # (B, 1)
            kelly_ext  = torch.cat([kelly, cash_score], dim=1)                     # (B, 8)
            raw = kelly_ext * direction * confidence
        else:
            raw = direction * confidence

        self._last_raw = raw
        return raw / (raw.sum(dim=1, keepdim=True) + 1e-8)

    def logits(self, state: torch.Tensor,
               kelly: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Return raw pre-normalisation scores as logits (for policy log_prob)."""
        self.forward(state, kelly)
        raw = self._last_raw
        # Convert to log-scale logits: log(raw) - mean(log(raw))
        log_raw = torch.log(raw.clamp(1e-8, 1.0))
        return log_raw - log_raw.mean(dim=-1, keepdim=True)


class CriticNetwork(nn.Module):
    def __init__(
        self,
        state_dim:   int = STATE_DIM,
        hidden_dims: List[int] = [512, 256, 128],
        dropout:     float = DROPOUT,
    ):
        super().__init__()
        layers = []
        prev = state_dim
        for h in hidden_dims:
            layers += [nn.Linear(prev, h), nn.LayerNorm(h), nn.GELU(), nn.Dropout(dropout)]
            prev = h
        layers.append(nn.Linear(prev, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, s): return self.net(s)
    def param_count(self): return sum(p.numel() for p in self.parameters())
