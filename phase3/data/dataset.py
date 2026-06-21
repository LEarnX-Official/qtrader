"""Phase 3 Dataset — multimodal features + targets."""

from typing import Optional, Tuple
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

from phase2.utils.config import TRAIN_END, VAL_END, META_CSV, Y_NPY
from phase3.utils.config import BATCH_SIZE_P3, NUM_WORKERS_P3
from phase3.utils.logging import get_logger

logger = get_logger(__name__)


class ObserverDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = torch.from_numpy(X).float()
        self.y = torch.from_numpy(y).float().unsqueeze(1)

    def __len__(self): return len(self.X)
    def __getitem__(self, i): return self.X[i], self.y[i]


def build_dataloaders(
    mm_features: np.ndarray,   # (N, 256)
    batch_size=BATCH_SIZE_P3,
    num_workers=NUM_WORKERS_P3,
) -> Tuple[DataLoader, DataLoader]:
    y    = np.load(Y_NPY)
    meta = pd.read_csv(META_CSV, parse_dates=["datetime"])

    train_mask = (meta["datetime"] <= TRAIN_END).values
    val_mask   = ((meta["datetime"] > TRAIN_END) & (meta["datetime"] <= VAL_END)).values

    def _loader(X, y, shuffle):
        return DataLoader(ObserverDataset(X, y), batch_size=batch_size,
                          shuffle=shuffle, num_workers=num_workers,
                          pin_memory=True, persistent_workers=num_workers > 0)

    train_loader = _loader(mm_features[train_mask], y[train_mask], True)
    val_loader   = _loader(mm_features[val_mask],   y[val_mask],   False)
    logger.info(f"Train: {train_mask.sum():,}  Val: {val_mask.sum():,}")
    return train_loader, val_loader
