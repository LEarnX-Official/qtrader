"""Phase 2 Pipeline — VAE training for crypto Ψ₀ estimation."""

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd
import torch

from phase2.analysis.analyzer import VAEAnalyzer
from phase2.data.dataset import MarketSequenceDataset, build_dataloaders
from phase2.models.vae import MarketVAE
from phase2.training.trainer import VAETrainer
from phase2.utils.config import (
    BATCH_SIZE, LATENT_DIM, LATENTS_ALL, LATENTS_TRAIN, LATENTS_VAL,
    META_CSV, MIN_ACTIVE_DIMS, MODELS_DIR, N_FEATURES, NUM_EPOCHS, NUM_WORKERS,
    RECON_THRESHOLD, SEQ_LEN, TRAIN_HISTORY, TRAIN_END, VAL_END,
    VAE_CKPT, VIZ_DIMS, VIZ_LATENT, X_NPY, Y_NPY,
)
from phase2.utils.logging import get_logger, setup_logging

logger = get_logger(__name__)


@dataclass
class Phase2Result:
    model:       MarketVAE
    latents_val: np.ndarray
    metrics:     Dict
    history:     Dict = field(default_factory=dict)
    device:      str  = "cpu"


class Phase2Pipeline:
    def __init__(self, num_epochs=NUM_EPOCHS, force_retrain=False, device=None):
        self.num_epochs    = num_epochs
        self.force_retrain = force_retrain
        self.device        = device or ("cuda" if torch.cuda.is_available() else "cpu")

    def run(self) -> Phase2Result:
        setup_logging()

        _banner("STEP 1  ─  Build DataLoaders  (train 2023–2024 | val 2025)")
        train_loader, val_loader, _ = build_dataloaders(BATCH_SIZE, NUM_WORKERS)

        meta  = pd.read_csv(META_CSV, parse_dates=["datetime"])
        y_all = np.load(Y_NPY)
        val_mask = (meta["datetime"] > TRAIN_END) & (meta["datetime"] <= VAL_END)
        y_val = y_all[val_mask.values]

        _banner("STEP 2  ─  Train / Load VAE")
        model, history = self._train_or_load(train_loader, val_loader)

        _banner("STEP 3  ─  Extract & Save Latents")
        self._save_latents(model)

        _banner("STEP 4  ─  Analysis")
        metrics, latents_val = VAEAnalyzer(model, self.device).run_all(
            train_loader, val_loader, y_val, history)

        _banner("PHASE 2 CHECKLIST")
        self._checklist(metrics)

        return Phase2Result(model=model, latents_val=latents_val,
                            metrics=metrics, history=history, device=self.device)

    def _train_or_load(self, train_loader, val_loader):
        MODELS_DIR.mkdir(parents=True, exist_ok=True)
        if VAE_CKPT.exists() and not self.force_retrain:
            logger.info(f"Loading existing checkpoint {VAE_CKPT}")
            _, model = VAETrainer.load_checkpoint(VAE_CKPT, self.device)
            history = json.load(open(TRAIN_HISTORY)) if TRAIN_HISTORY.exists() else {}
            return model, history
        model   = MarketVAE(SEQ_LEN, N_FEATURES, LATENT_DIM)
        logger.info(f"MarketVAE params: {model.param_count():,}")
        trainer = VAETrainer(model, self.device)
        history = trainer.train(train_loader, val_loader, self.num_epochs, VAE_CKPT)
        return model, history

    def _save_latents(self, model):
        from torch.utils.data import DataLoader as DL

        analyzer = VAEAnalyzer(model, self.device)
        meta = pd.read_csv(META_CSV, parse_dates=["datetime"])
        X_all = np.load(X_NPY, mmap_mode="r")

        def _extract(mask, path):
            path.parent.mkdir(parents=True, exist_ok=True)
            chunk   = np.array(X_all[mask])
            loader  = DL(MarketSequenceDataset(chunk), batch_size=BATCH_SIZE,
                         shuffle=False, num_workers=NUM_WORKERS)
            latents = analyzer.extract_latents(loader)
            np.save(path, latents)
            logger.info(f"Saved {path}  {latents.shape}")
            return latents

        train_mask = (meta["datetime"] <= TRAIN_END).values
        val_mask   = ((meta["datetime"] > TRAIN_END) & (meta["datetime"] <= VAL_END)).values

        _extract(train_mask, LATENTS_TRAIN)
        _extract(val_mask,   LATENTS_VAL)

        # Full-dataset latents (all dates) for Phase 3+
        full_loader = DL(MarketSequenceDataset(np.array(X_all)), batch_size=BATCH_SIZE,
                         shuffle=False, num_workers=NUM_WORKERS)
        la = analyzer.extract_latents(full_loader)
        np.save(LATENTS_ALL, la)
        logger.info(f"Saved {LATENTS_ALL}  {la.shape}")

    @staticmethod
    def load_result(device="cpu") -> Optional[Phase2Result]:
        if not VAE_CKPT.exists():
            logger.warning("No VAE checkpoint found")
            return None
        _, model = VAETrainer.load_checkpoint(VAE_CKPT, device)
        lv = np.load(LATENTS_VAL) if LATENTS_VAL.exists() else None
        history = json.load(open(TRAIN_HISTORY)) if TRAIN_HISTORY.exists() else {}
        return Phase2Result(model=model, latents_val=lv, metrics={},
                            history=history, device=device)

    @staticmethod
    def _checklist(metrics):
        checks = {
            f"Recon error < {RECON_THRESHOLD} ({metrics.get('val_recon_loss',999):.4f})":
                metrics.get("val_recon_loss", 999) < RECON_THRESHOLD,
            f"KL > 0 (no collapse): {metrics.get('val_kl',0):.4f}":
                metrics.get("val_kl", 0) > 0.001,
            f"PCA var > 30%: {metrics.get('pca_explained_var',0):.1%}":
                metrics.get("pca_explained_var", 0) > 0.30,
            f"Active latent dims ≥ {MIN_ACTIVE_DIMS} ({metrics.get('active_latent_dims',0)})":
                metrics.get("active_latent_dims", 0) >= MIN_ACTIVE_DIMS,
            "vae_best.pt saved":                VAE_CKPT.exists(),
            "latent_val.npy saved":             LATENTS_VAL.exists(),
            "latent_representations.npy saved": LATENTS_ALL.exists(),
        }
        all_ok = True
        for label, passed in checks.items():
            logger.info(f"  [{'✓' if passed else '✗'}] {label}")
            if not passed: all_ok = False
        if all_ok:
            logger.info("Phase 2 COMPLETE — ready for Phase 3 (Transformer / ΣAᵢ)")
        else:
            logger.warning("Phase 2 finished with failures")


def _banner(t): logger.info("="*60 + f"\n  {t}\n" + "="*60)
