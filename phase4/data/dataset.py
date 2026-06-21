"""Phase 4 Dataset — (Ψ₀, ΣAᵢ, α) → bin label."""

from typing import Tuple
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

from phase4.utils.config import (
    BATCH_SIZE_P4, BIN_EDGES, META_CSV, NUM_WORKERS_P4,
    TRAIN_END, VAL_END, Y_NPY,
    LATENTS_ALL, SIGMA_AI_NPY, ALPHA_NPY,
)
from phase4.utils.logging import get_logger

logger = get_logger(__name__)


def returns_to_bins(y: np.ndarray, bin_edges: np.ndarray) -> np.ndarray:
    """Convert continuous returns to bin indices (0..NUM_BINS-1)."""
    bins = np.digitize(y, bin_edges[1:-1])   # interior edges only
    return np.clip(bins, 0, len(bin_edges)-2).astype(np.int64)


class PINNDataset(Dataset):
    def __init__(self, psi0: np.ndarray, sigma: np.ndarray,
                 alpha: np.ndarray, y_bins: np.ndarray, y_cont: np.ndarray):
        self.psi0   = torch.from_numpy(psi0).float()
        self.sigma  = torch.from_numpy(sigma).float()
        self.alpha  = torch.from_numpy(alpha).float()
        self.y_bins = torch.from_numpy(y_bins).long()
        self.y_cont = torch.from_numpy(y_cont).float()

    def __len__(self): return len(self.psi0)
    def __getitem__(self, i):
        return (self.psi0[i], self.sigma[i], self.alpha[i],
                self.y_bins[i], self.y_cont[i])


def build_dataloaders(batch_size=BATCH_SIZE_P4, num_workers=NUM_WORKERS_P4):
    psi0  = np.load(LATENTS_ALL)
    sigma = np.load(SIGMA_AI_NPY)
    alpha = np.load(ALPHA_NPY)
    y     = np.load(Y_NPY)
    meta  = pd.read_csv(META_CSV, parse_dates=["datetime"])

    y_bins = returns_to_bins(y, BIN_EDGES)

    train_m = (meta["datetime"] <= TRAIN_END).values
    val_m   = ((meta["datetime"] > TRAIN_END) & (meta["datetime"] <= VAL_END)).values

    def _loader(m, shuffle):
        ds = PINNDataset(psi0[m], sigma[m], alpha[m], y_bins[m], y[m])
        return DataLoader(ds, batch_size=batch_size, shuffle=shuffle,
                          num_workers=num_workers, pin_memory=True,
                          persistent_workers=num_workers > 0)

    logger.info(f"PINN Train: {train_m.sum():,}  Val: {val_m.sum():,}")
    return _loader(train_m, True), _loader(val_m, False)
