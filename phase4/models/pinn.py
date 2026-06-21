"""
Born-Rule PINN — identical to crypto/phase4 but imports local config.
P(price_k | t) = |⟨ψ_k | A(O,t) | Ψ₀⟩|²
"""

from typing import Tuple
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F

from phase4.utils.config import ALPHA_DIM, DROPOUT, NUM_BINS, PSI0_DIM, SIGMA_AI_DIM

HILBERT_DIM   = 64
OPERATOR_RANK = 16


class BornRulePINN(nn.Module):
    def __init__(self, psi0_dim=PSI0_DIM, sigma_ai_dim=SIGMA_AI_DIM,
                 num_bins=NUM_BINS, hilbert_dim=HILBERT_DIM,
                 operator_rank=OPERATOR_RANK, dropout=DROPOUT):
        super().__init__()
        self.hilbert_dim = hilbert_dim
        self.num_bins    = num_bins

        # Learnable bin eigenstates
        self.bin_states = nn.Parameter(torch.empty(num_bins, hilbert_dim))
        nn.init.orthogonal_(self.bin_states)

        # Step 1: Ψ₀ → Hilbert
        self.psi0_proj = nn.Sequential(
            nn.Linear(psi0_dim, 64), nn.GELU(),
            nn.Linear(64, hilbert_dim),
        )

        # Step 2: ΣAᵢ → low-rank operator factors U, V ∈ ℝ^(H×R)
        self.op_U = nn.Sequential(
            nn.Linear(sigma_ai_dim, 256), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(256, hilbert_dim * operator_rank),
        )
        self.op_V = nn.Sequential(
            nn.Linear(sigma_ai_dim, 256), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(256, hilbert_dim * operator_rank),
        )

        # Alignment gate: α modulates temperature
        self.alpha_gate = nn.Linear(ALPHA_DIM, 1)

    def forward(self, psi0: torch.Tensor, sigma_ai: torch.Tensor,
                alpha: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        B = psi0.size(0)
        H = self.hilbert_dim
        R = self.op_U[0].out_features // H if False else 16  # OPERATOR_RANK

        # Step 1: project and normalise Ψ₀
        psi_h = self.psi0_proj(psi0)                        # (B, H)
        psi_h = F.normalize(psi_h, p=2, dim=1)

        # Step 2: construct low-rank operator A = U @ Vᵀ  (B, H, H)
        U = self.op_U(sigma_ai).view(B, H, R)               # (B, H, R)
        V = self.op_V(sigma_ai).view(B, H, R)               # (B, H, R)
        A = torch.bmm(U, V.transpose(1,2)) / R              # (B, H, H)

        # Step 3: evolve |ψ'⟩ = A|Ψ₀⟩
        psi_evolved = torch.bmm(A, psi_h.unsqueeze(2)).squeeze(2)   # (B, H)

        # Step 4: inner products ⟨ψ_k | ψ'⟩
        # bin_states: (K, H) → (B, K, H) × (B, H, 1) → (B, K)
        inner = torch.matmul(
            self.bin_states.unsqueeze(0).expand(B,-1,-1),    # (B, K, H)
            psi_evolved.unsqueeze(2)                          # (B, H, 1)
        ).squeeze(2)                                          # (B, K)

        # Step 5: Born rule P_k = |⟨ψ_k|ψ'⟩|²
        probs_raw = inner ** 2                               # (B, K)

        # Alignment modulates temperature (high α → sharper distribution)
        temp = 1.0 / (torch.sigmoid(self.alpha_gate(alpha)) + 0.1)   # (B,1)
        probs = F.softmax(probs_raw * temp, dim=1)           # (B, K)

        # Expected return and variance
        from phase4.utils.config import BIN_CENTERS
        bc  = torch.tensor(BIN_CENTERS, dtype=torch.float32, device=probs.device)
        E   = (probs * bc).sum(dim=1)                        # (B,)
        V   = (probs * (bc - E.unsqueeze(1))**2).sum(dim=1) # (B,)

        return probs, E, V

    def param_count(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
