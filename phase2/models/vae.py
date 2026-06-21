"""MarketVAE — identical to crypto/phase2/models/vae.py, imports local encoder/decoder."""

from dataclasses import dataclass
from typing import Tuple
import torch, torch.nn as nn, torch.nn.functional as F

from phase2.models.encoder import MarketStateEncoder
from phase2.models.decoder import MarketStateDecoder
from phase2.utils.config import LATENT_DIM, N_FEATURES, SEQ_LEN


@dataclass
class VAEOutput:
    x_recon: torch.Tensor
    mu:      torch.Tensor
    logvar:  torch.Tensor
    z:       torch.Tensor


class MarketVAE(nn.Module):
    def __init__(self, seq_len=SEQ_LEN, n_features=N_FEATURES, latent_dim=LATENT_DIM):
        super().__init__()
        self.latent_dim = latent_dim
        self.encoder    = MarketStateEncoder(seq_len, n_features, latent_dim)
        self.decoder    = MarketStateDecoder(latent_dim, seq_len, n_features)

    def reparameterize(self, mu, logvar):
        return mu + torch.exp(0.5 * logvar) * torch.randn_like(mu)

    def forward(self, x):
        mu, logvar = self.encoder(x)
        z          = self.reparameterize(mu, logvar)
        return VAEOutput(x_recon=self.decoder(z), mu=mu, logvar=logvar, z=z)

    @torch.no_grad()
    def encode(self, x):
        mu, _ = self.encoder(x)
        return mu

    def param_count(self):
        return sum(p.numel() for p in self.parameters())


@dataclass
class VAELoss:
    total: torch.Tensor
    recon: torch.Tensor
    kl:    torch.Tensor
    beta:  float


def vae_loss(x, output: VAEOutput, beta=0.01) -> VAELoss:
    recon = F.mse_loss(output.x_recon, x, reduction="mean")
    kl    = -0.5 * torch.sum(1 + output.logvar - output.mu.pow(2) - output.logvar.exp()) / x.size(0)
    return VAELoss(total=recon + beta * kl, recon=recon, kl=kl, beta=beta)
