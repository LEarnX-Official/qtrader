"""Phase 3 Pipeline — Observer Aggregator Transformer for crypto ΣAᵢ."""

import json
from dataclasses import dataclass, field
from typing import Dict, Optional

import numpy as np
import pandas as pd
import torch

from phase1.utils.config import META_CSV, X_NPY, Y_NPY
from phase2.utils.config import LATENTS_ALL, LATENTS_TRAIN, LATENTS_VAL, TRAIN_END, VAL_END
from phase3.data.builder import build_multimodal_features
from phase3.data.dataset import build_dataloaders
from phase3.models.transformer import ObserverAggregatorTransformer
from phase3.training.trainer import ObserverTrainer
from phase3.utils.config import (
    ALPHA_NPY, ALPHA_TRAIN, ALPHA_VAL,
    BATCH_SIZE_P3, DATA_P3, MIN_DIRECTION_ACC,
    MODELS_DIR, NUM_EPOCHS_P3, NUM_WORKERS_P3,
    OBS_CKPT, OBS_HISTORY, SIGMA_AI_NPY,
    SIGMA_AI_TRAIN, SIGMA_AI_VAL,
)
from phase3.utils.logging import get_logger, setup_logging

logger = get_logger(__name__)


@dataclass
class Phase3Result:
    model:        ObserverAggregatorTransformer
    sigma_ai_val: np.ndarray    # (N_val, 512)
    alpha_val:    np.ndarray    # (N_val, 1)
    metrics:      Dict
    history:      Dict = field(default_factory=dict)
    device:       str  = "cpu"


class Phase3Pipeline:
    def __init__(self, num_epochs=NUM_EPOCHS_P3, force_retrain=False, device=None):
        self.num_epochs    = num_epochs
        self.force_retrain = force_retrain
        self.device        = device or ("cuda" if torch.cuda.is_available() else "cpu")

    def run(self) -> Phase3Result:
        setup_logging()

        _banner("STEP 1  ─  Load latents + build multimodal features  (train 2023–2024 | val 2025)")
        mm_all, meta = self._build_mm_features()

        _banner("STEP 2  ─  Build DataLoaders")
        train_loader, val_loader = build_dataloaders(mm_all)

        _banner("STEP 3  ─  Train / Load Transformer")
        model, history = self._train_or_load(train_loader, val_loader)

        _banner("STEP 4  ─  Extract ΣAᵢ + α for train & val")
        self._extract_all(model, mm_all, meta)

        _banner("PHASE 3 CHECKLIST")
        val_acc = max(history.get("val_acc", [0]))
        metrics = {"val_direction_acc": val_acc}
        self._checklist(metrics)

        return Phase3Result(
            model=model,
            sigma_ai_val=np.load(SIGMA_AI_VAL),
            alpha_val=np.load(ALPHA_VAL),
            metrics=metrics, history=history, device=self.device,
        )

    def _build_mm_features(self):
        meta    = pd.read_csv(META_CSV, parse_dates=["datetime"])
        latents = np.load(LATENTS_ALL)
        X_all   = np.load(X_NPY, mmap_mode="r")
        mm = build_multimodal_features(latents, X_all, meta)
        DATA_P3.mkdir(parents=True, exist_ok=True)
        np.save(DATA_P3 / "multimodal_features.npy", mm)
        logger.info(f"Multimodal features: {mm.shape}")
        return mm, meta

    def _train_or_load(self, train_loader, val_loader):
        MODELS_DIR.mkdir(parents=True, exist_ok=True)
        if OBS_CKPT.exists() and not self.force_retrain:
            logger.info(f"Loading existing checkpoint {OBS_CKPT}")
            t, model = ObserverTrainer.load_checkpoint(OBS_CKPT, self.device)
            history = json.load(open(OBS_HISTORY)) if OBS_HISTORY.exists() else {}
            return model, history
        model   = ObserverAggregatorTransformer()
        logger.info(f"Transformer params: {model.param_count():,}")
        trainer = ObserverTrainer(model, self.device)
        history = trainer.train(train_loader, val_loader, self.num_epochs, OBS_CKPT)
        return model, history

    @torch.no_grad()
    def _extract_all(self, model, mm_all, meta):
        from torch.utils.data import DataLoader, TensorDataset
        model.eval()
        loader = DataLoader(
            TensorDataset(torch.from_numpy(mm_all).float()),
            batch_size=BATCH_SIZE_P3, shuffle=False, num_workers=NUM_WORKERS_P3)

        all_sigma, all_alpha = [], []
        for (x,) in loader:
            s, a = model(x.to(self.device))
            all_sigma.append(s.cpu().numpy())
            all_alpha.append(a.cpu().numpy())
        sigma = np.concatenate(all_sigma)
        alpha = np.concatenate(all_alpha)

        np.save(SIGMA_AI_NPY, sigma); np.save(ALPHA_NPY, alpha)
        logger.info(f"Saved {SIGMA_AI_NPY} {sigma.shape}")

        train_mask = (meta["datetime"] <= TRAIN_END).values
        val_mask   = ((meta["datetime"] > TRAIN_END) & (meta["datetime"] <= VAL_END)).values

        for mask, sp, ap in [(train_mask, SIGMA_AI_TRAIN, ALPHA_TRAIN),
                              (val_mask,   SIGMA_AI_VAL,   ALPHA_VAL)]:
            np.save(sp, sigma[mask]); np.save(ap, alpha[mask])
            logger.info(f"  {sp.name}: {mask.sum():,} rows")

    @staticmethod
    def load_result(device="cpu") -> Optional[Phase3Result]:
        if not OBS_CKPT.exists():
            logger.warning("No observer checkpoint found")
            return None
        _, model = ObserverTrainer.load_checkpoint(OBS_CKPT, device)
        history  = json.load(open(OBS_HISTORY)) if OBS_HISTORY.exists() else {}
        return Phase3Result(
            model=model,
            sigma_ai_val=np.load(SIGMA_AI_VAL) if SIGMA_AI_VAL.exists() else None,
            alpha_val=np.load(ALPHA_VAL)        if ALPHA_VAL.exists()    else None,
            metrics={}, history=history, device=device,
        )

    @staticmethod
    def _checklist(metrics):
        acc = metrics.get("val_direction_acc", 0)
        checks = {
            f"Direction acc > {MIN_DIRECTION_ACC:.0%}  (acc={acc:.4f})": acc > MIN_DIRECTION_ACC,
            "observer_transformer_best.pt saved": OBS_CKPT.exists(),
            "sigma_ai.npy saved":                 SIGMA_AI_NPY.exists(),
            "alignment_scores.npy saved":         ALPHA_NPY.exists(),
        }
        all_ok = True
        for lbl, ok in checks.items():
            logger.info(f"  [{'✓' if ok else '✗'}] {lbl}")
            if not ok: all_ok = False
        if all_ok: logger.info("Phase 3 COMPLETE — ready for Phase 4 (PINN)")
        else:      logger.warning("Phase 3 finished with failures")


def _banner(t): logger.info("="*60 + f"\n  {t}\n" + "="*60)
