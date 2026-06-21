"""Decoder p_θ(X_t|z) — symmetric to encoder."""

import torch, torch.nn as nn, torch.nn.functional as F
from phase2.utils.config import LATENT_DIM, N_FEATURES, SEQ_LEN


class MarketStateDecoder(nn.Module):
    def __init__(self, latent_dim=LATENT_DIM, seq_len=SEQ_LEN, n_features=N_FEATURES):
        super().__init__()
        self.seq_comp = seq_len // 4

        self.fc1     = nn.Linear(latent_dim, 256)
        self.fc2     = nn.Linear(256, 64 * self.seq_comp)
        self.deconv1 = nn.ConvTranspose1d(64, 128, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.ln1     = nn.LayerNorm([128, seq_len // 2])
        self.deconv2 = nn.ConvTranspose1d(128, 256, kernel_size=5, stride=2, padding=2, output_padding=1)
        self.ln2     = nn.LayerNorm([256, seq_len])
        self.deconv3 = nn.ConvTranspose1d(256, n_features, kernel_size=5, stride=1, padding=2)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        h = F.gelu(self.fc1(z))
        h = F.gelu(self.fc2(h)).reshape(h.size(0) if False else z.size(0), 64, self.seq_comp)
        h = F.gelu(self.ln1(self.deconv1(h)))
        h = F.gelu(self.ln2(self.deconv2(h)))
        return self.deconv3(h).transpose(1, 2)
