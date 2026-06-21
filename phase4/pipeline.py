"""Phase 4 Pipeline — Born Rule PINN for crypto collapse probabilities."""

import json
from dataclasses import dataclass, field
from typing import Dict, Optional

import numpy as np
import pandas as pd
import torch

from phase1.utils.config import META_CSV, Y_NPY
from phase2.utils.config import LATENTS_ALL, TRAIN_END, VAL_END
from phase3.utils.config import ALPHA_NPY, SIGMA_AI_NPY
from phase4.data.dataset import build_dataloaders
from phase4.models.pinn import BornRulePINN
from phase4.training.trainer import PINNTrainer
from phase4.utils.config import (
    BATCH_SIZE_P4, COLLAPSE_PROBS, COLLAPSE_TRAIN, COLLAPSE_VAL,
    DATA_P4, EXPECTED_RETS, MIN_DIRECTION_ACC_P4,
    MODELS_DIR, NUM_EPOCHS_P4, NUM_WORKERS_P4,
    PINN_CKPT, PINN_HISTORY, RETURN_VAR,
)
from phase4.utils.logging import get_logger, setup_logging

logger = get_logger(__name__)


@dataclass
class Phase4Result:
    model:        BornRulePINN
    collapse_val: np.ndarray   # (N_val, K)
    exp_ret_val:  np.ndarray   # (N_val,)
    metrics:      Dict
    history:      Dict = field(default_factory=dict)
    device:       str  = "cpu"


class Phase4Pipeline:
    def __init__(self, num_epochs=NUM_EPOCHS_P4, force_retrain=False, device=None):
        self.num_epochs    = num_epochs
        self.force_retrain = force_retrain
        self.device        = device or ("cuda" if torch.cuda.is_available() else "cpu")

    def run(self) -> Phase4Result:
        setup_logging()

        _banner("STEP 1  ─  Build DataLoaders  (train 2023–2024 | val 2025)")
        train_loader, val_loader = build_dataloaders()

        _banner("STEP 2  ─  Train / Load PINN")
        model, history = self._train_or_load(train_loader, val_loader)

        _banner("STEP 3  ─  Extract collapse probabilities for train & val")
        self._extract_all(model)

        _banner("PHASE 4 CHECKLIST")
        val_acc = max(history.get("val_acc", [0]))
        metrics = {"val_direction_acc": val_acc}
        self._checklist(metrics)

        return Phase4Result(
            model=model,
            collapse_val=np.load(COLLAPSE_VAL),
            exp_ret_val=np.load(EXPECTED_RETS),
            metrics=metrics, history=history, device=self.device,
        )

    def _train_or_load(self, train_loader, val_loader):
        MODELS_DIR.mkdir(parents=True, exist_ok=True)
        if PINN_CKPT.exists() and not self.force_retrain:
            logger.info(f"Loading existing checkpoint {PINN_CKPT}")
            _, model = PINNTrainer.load_checkpoint(PINN_CKPT, self.device)
            history = json.load(open(PINN_HISTORY)) if PINN_HISTORY.exists() else {}
            return model, history
        model   = BornRulePINN()
        logger.info(f"PINN params: {model.param_count():,}")
        trainer = PINNTrainer(model, self.device)
        history = trainer.train(train_loader, val_loader, self.num_epochs, PINN_CKPT)
        return model, history

    @torch.no_grad()
    def _extract_all(self, model):
        from torch.utils.data import DataLoader, TensorDataset
        DATA_P4.mkdir(parents=True, exist_ok=True)
        model.eval()

        psi0  = np.load(LATENTS_ALL)
        sigma = np.load(SIGMA_AI_NPY)
        alpha = np.load(ALPHA_NPY)
        y     = np.load(Y_NPY)
        meta  = pd.read_csv(META_CSV, parse_dates=["datetime"])

        loader = torch.utils.data.DataLoader(
            TensorDataset(torch.from_numpy(psi0).float(),
                          torch.from_numpy(sigma).float(),
                          torch.from_numpy(alpha).float()),
            batch_size=BATCH_SIZE_P4, shuffle=False, num_workers=NUM_WORKERS_P4)

        all_probs, all_E, all_V = [], [], []
        for p0, sg, al in loader:
            probs, E, V = model(p0.to(self.device), sg.to(self.device), al.to(self.device))
            all_probs.append(probs.cpu().numpy())
            all_E.append(E.cpu().numpy())
            all_V.append(V.cpu().numpy())

        probs_all = np.concatenate(all_probs)
        E_all     = np.concatenate(all_E)
        V_all     = np.concatenate(all_V)

        np.save(COLLAPSE_PROBS, probs_all)
        np.save(EXPECTED_RETS,  E_all)
        np.save(RETURN_VAR,     V_all)
        logger.info(f"Saved {COLLAPSE_PROBS}  {probs_all.shape}")

        train_m = (meta["datetime"] <= TRAIN_END).values
        val_m   = ((meta["datetime"] > TRAIN_END) & (meta["datetime"] <= VAL_END)).values

        np.save(COLLAPSE_TRAIN, probs_all[train_m])
        np.save(COLLAPSE_VAL,   probs_all[val_m])
        logger.info(f"Train: {train_m.sum():,}  Val: {val_m.sum():,}")

    @staticmethod
    def load_result(device="cpu") -> Optional[Phase4Result]:
        if not PINN_CKPT.exists():
            logger.warning("No PINN checkpoint found")
            return None
        _, model = PINNTrainer.load_checkpoint(PINN_CKPT, device)
        history  = json.load(open(PINN_HISTORY)) if PINN_HISTORY.exists() else {}
        return Phase4Result(
            model=model,
            collapse_val=np.load(COLLAPSE_VAL) if COLLAPSE_VAL.exists() else None,
            exp_ret_val=np.load(EXPECTED_RETS)  if EXPECTED_RETS.exists()  else None,
            metrics={}, history=history, device=device,
        )

    @staticmethod
    def _checklist(metrics):
        acc = metrics.get("val_direction_acc", 0)
        checks = {
            f"Direction acc > {MIN_DIRECTION_ACC_P4:.0%} (acc={acc:.4f})": acc > MIN_DIRECTION_ACC_P4,
            "pinn_collapse_best.pt saved":  PINN_CKPT.exists(),
            "collapse_probs.npy saved":     COLLAPSE_PROBS.exists(),
            "expected_returns.npy saved":   EXPECTED_RETS.exists(),
        }
        all_ok = True
        for lbl, ok in checks.items():
            logger.info(f"  [{'✓' if ok else '✗'}] {lbl}")
            if not ok: all_ok = False
        if all_ok: logger.info("Phase 4 COMPLETE — ready for Phase 5 (PPO)")
        else:      logger.warning("Phase 4 finished with failures")


def _banner(t): logger.info("="*60 + f"\n  {t}\n" + "="*60)
