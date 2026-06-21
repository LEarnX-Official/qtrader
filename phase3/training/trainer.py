"""
Phase 3 Trainer — Observer Aggregator with BCE direction loss.

Root cause of previous failure:
    MSE(pred_return, true_return) collapses to predicting ~0 for all inputs
    because the mean return ≈ 0 and variance is tiny (~6e-5).
    The model found the trivial solution: predict 0, get MSE = Var(y) = 0.00008.
    Val direction accuracy stayed at 50% (random) throughout.

Fix:
    Replace MSE on raw return with BCE on sign(return).
    y_sign = 1 if return > 0 else 0
    dir_loss = BCE(sigmoid(σAᵢ → logit), y_sign)

    BCE directly optimises direction accuracy, not return magnitude.
    The model can no longer collapse to predicting 0.
    A balanced class weight handles any up/down imbalance in crypto.
"""

import json
from pathlib import Path
from typing import Dict, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from phase3.models.transformer import ObserverAggregatorTransformer
from phase3.utils.config import (
    ALIGN_LOSS_WEIGHT, BATCH_SIZE_P3, GRAD_CLIP_P3, LEARNING_RATE_P3,
    MODELS_DIR, NUM_EPOCHS_P3, OBS_CKPT, OBS_HISTORY, SIGMA_DIM,
    WEIGHT_DECAY_P3,
)
from phase3.utils.logging import get_logger

logger = get_logger(__name__)


class ObserverTrainer:
    def __init__(self, model: ObserverAggregatorTransformer, device="cpu"):
        self.model    = model.to(device)
        self.device   = device

        # Direction head: σAᵢ (512) → logit (1) — predicts P(up) via sigmoid
        # Replaced from Linear → return-value predictor
        #                    to Linear → direction logit (BCE optimised)
        self.dir_head = nn.Linear(SIGMA_DIM, 1).to(device)

        self.opt = torch.optim.AdamW(
            list(model.parameters()) + list(self.dir_head.parameters()),
            lr=LEARNING_RATE_P3, weight_decay=WEIGHT_DECAY_P3)
        self.sched = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.opt, T_max=NUM_EPOCHS_P3, eta_min=1e-6)
        self.history: Dict = {k: [] for k in
            ["train_loss", "train_dir", "train_align",
             "val_loss",   "val_dir",   "val_align",   "val_acc"]}

    def _run_epoch(self, loader: DataLoader, train: bool = True):
        self.model.train(train)
        self.dir_head.train(train)
        totals  = [0., 0., 0.]
        n       = 0
        correct = total = 0

        # Compute class weight on first train epoch call to balance up/down
        # (crypto returns are ~symmetric but let's be safe)
        pos_weight = None

        ctx = torch.enable_grad() if train else torch.no_grad()
        with ctx:
            for x, y in loader:
                x = x.to(self.device)
                y = y.to(self.device)          # (B, 1)  raw log-return

                # Convert continuous return → binary direction label
                y_sign = (y > 0).float()       # 1 = up, 0 = down/flat

                sigma, alpha = self.model(x)
                logit        = self.dir_head(sigma)    # (B, 1) unbounded

                # BCE with logits — numerically stable, directly optimises accuracy
                # pos_weight balances up vs down if dataset is skewed
                if train and pos_weight is None:
                    n_pos = y_sign.sum().clamp(min=1)
                    n_neg = (1 - y_sign).sum().clamp(min=1)
                    pos_weight = (n_neg / n_pos).detach()

                dir_loss = F.binary_cross_entropy_with_logits(
                    logit, y_sign,
                    pos_weight=pos_weight if train else None,
                )

                # Alignment auxiliary loss — keep as MSE on |return|
                # (measures whether modality agreement predicts move magnitude)
                abs_r      = y.abs()
                align_loss = F.mse_loss(alpha, abs_r / (abs_r.max() + 1e-8))

                loss = dir_loss + ALIGN_LOSS_WEIGHT * align_loss

                if train:
                    self.opt.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(
                        list(self.model.parameters()) +
                        list(self.dir_head.parameters()),
                        GRAD_CLIP_P3,
                    )
                    self.opt.step()

                totals[0] += loss.item()
                totals[1] += dir_loss.item()
                totals[2] += align_loss.item()
                n         += 1

                if not train:
                    pred_up  = (logit > 0)        # sigmoid(logit) > 0.5 ↔ logit > 0
                    true_up  = (y > 0)
                    correct += (pred_up == true_up).sum().item()
                    total   += len(y)

        avg = [t / n for t in totals]
        acc = correct / total if (not train and total > 0) else 0.0
        return avg[0], avg[1], avg[2], acc

    def train(
        self,
        train_loader: DataLoader,
        val_loader:   DataLoader,
        num_epochs:   int  = NUM_EPOCHS_P3,
        ckpt_path:    Path = OBS_CKPT,
    ) -> Dict:
        ckpt_path = Path(ckpt_path)
        ckpt_path.parent.mkdir(parents=True, exist_ok=True)
        best_acc = 0.0

        for ep in range(1, num_epochs + 1):
            tl, td, ta, _    = self._run_epoch(train_loader, train=True)
            vl, vd, va, acc  = self._run_epoch(val_loader,   train=False)
            self.sched.step()

            logger.info(
                f"Ep {ep:3d}/{num_epochs} | "
                f"train loss={tl:.5f} dir={td:.5f} align={ta:.5f} | "
                f"val loss={vl:.5f} dir={vd:.5f} align={va:.5f} acc={acc:.4f}"
            )

            for k, v in zip(self.history, [tl, td, ta, vl, vd, va, acc]):
                self.history[k].append(v)

            if acc > best_acc:
                best_acc = acc
                torch.save({
                    "epoch":                ep,
                    "model_state_dict":     self.model.state_dict(),
                    "dir_head_state_dict":  self.dir_head.state_dict(),
                    "val_acc":              acc,
                    "history":              self.history,
                }, ckpt_path)
                logger.info(f"  ✓ Best saved (acc={acc:.4f})")

        with open(OBS_HISTORY, "w") as f:
            json.dump(self.history, f, indent=2)
        return self.history

    @staticmethod
    def load_checkpoint(ckpt_path: Path = OBS_CKPT, device: str = "cpu"):
        ckpt  = torch.load(ckpt_path, map_location=device, weights_only=False)
        model = ObserverAggregatorTransformer()
        model.load_state_dict(ckpt["model_state_dict"])
        model.to(device)
        t = ObserverTrainer(model, device)
        t.dir_head.load_state_dict(ckpt["dir_head_state_dict"])
        t.history = ckpt.get("history", t.history)
        logger.info(
            f"Loaded observer ckpt (ep={ckpt['epoch']}, acc={ckpt['val_acc']:.4f})"
        )
        return t, model
