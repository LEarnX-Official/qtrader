"""Encoder q_φ(z|X_t) — identical architecture to crypto/phase2, adapted dims."""

from typing import Tuple
import torch, torch.nn as nn, torch.nn.functional as F
from phase2.utils.config import LATENT_DIM, N_FEATURES, SEQ_LEN


class MarketStateEncoder(nn.Module):
    def __init__(self, seq_len=SEQ_LEN, n_features=N_FEATURES, latent_dim=LATENT_DIM):
        super().__init__()
        self.seq_len    = seq_len
        self.n_features = n_features
        self.latent_dim = latent_dim

        self.conv1 = nn.Conv1d(n_features, 256, kernel_size=5, stride=1, padding=2)
        self.ln1   = nn.LayerNorm([256, seq_len])
        self.conv2 = nn.Conv1d(256, 128, kernel_size=5, stride=2, padding=2)
        self.ln2   = nn.LayerNorm([128, seq_len // 2])
        self.conv3 = nn.Conv1d(128,  64, kernel_size=3, stride=2, padding=1)
        self.ln3   = nn.LayerNorm([ 64, seq_len // 4])

        self.dropout = nn.Dropout(0.2)
        self.fc1     = nn.Linear(64 * (seq_len // 4), 256)
        self.fc_mu     = nn.Linear(256, latent_dim)
        self.fc_logvar = nn.Linear(256, latent_dim)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h = x.transpose(1, 2)
        h = F.gelu(self.ln1(self.conv1(h)))
        h = F.gelu(self.ln2(self.conv2(h)))
        h = F.gelu(self.ln3(self.conv3(h)))
        h = self.dropout(h.reshape(h.size(0), -1))
        h = F.gelu(self.fc1(h))
        return self.fc_mu(h), self.fc_logvar(h)
