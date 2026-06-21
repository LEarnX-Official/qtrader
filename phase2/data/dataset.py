"""
Phase 2 Dataset — train / val splits.

  train : 2023-01-01 → 2024-12-31
  val   : 2025-01-01 → 2025-12-31
"""

from typing import Optional, Tuple
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

from phase2.utils.config import (
    BATCH_SIZE, NUM_WORKERS, X_NPY, Y_NPY, META_CSV,
    TRAIN_END, VAL_END,
)
from phase2.utils.logging import get_logger

logger = get_logger(__name__)


class MarketSequenceDataset(Dataset):
    def __init__(self, X: np.ndarray, y: Optional[np.ndarray] = None):
        self.X = torch.from_numpy(X).float()
        self.y = torch.from_numpy(y).float() if y is not None else None

    def __len__(self): return len(self.X)

    def __getitem__(self, idx):
        return (self.X[idx], self.y[idx]) if self.y is not None else self.X[idx]


def build_dataloaders(
    batch_size: int = BATCH_SIZE,
    num_workers: int = NUM_WORKERS,
) -> Tuple[DataLoader, DataLoader, int]:
    logger.info(f"Loading sequences from {X_NPY}")
    X    = np.load(X_NPY, mmap_mode="r")
    y    = np.load(Y_NPY)
    meta = pd.read_csv(META_CSV, parse_dates=["datetime"])

    logger.info(f"X:{X.shape}  y:{y.shape}")

    train_mask = meta["datetime"] <= TRAIN_END
    val_mask   = (meta["datetime"] > TRAIN_END) & (meta["datetime"] <= VAL_END)

    X_train = np.array(X[train_mask.values])
    X_val   = np.array(X[val_mask.values])
    y_train = y[train_mask.values]
    y_val   = y[val_mask.values]

    logger.info(
        f"Train: {len(X_train):,} "
        f"({meta.loc[train_mask,'datetime'].min().date()} → "
        f"{meta.loc[train_mask,'datetime'].max().date()})"
    )
    logger.info(
        f"Val:   {len(X_val):,} "
        f"({meta.loc[val_mask,'datetime'].min().date()} → "
        f"{meta.loc[val_mask,'datetime'].max().date()})"
    )

    def _loader(ds, shuffle):
        return DataLoader(ds, batch_size=batch_size, shuffle=shuffle,
                          num_workers=num_workers, pin_memory=True,
                          persistent_workers=num_workers > 0)

    return (
        _loader(MarketSequenceDataset(X_train, y_train), True),
        _loader(MarketSequenceDataset(X_val,   y_val),   False),
        X.shape[2],
    )
