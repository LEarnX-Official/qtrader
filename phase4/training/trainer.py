"""Phase 4 Trainer — Born Rule PINN with CE + physics loss."""

import json
from pathlib import Path
from typing import Dict, Tuple
import torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import DataLoader

from phase4.models.pinn import BornRulePINN
from phase4.utils.config import (
    BATCH_SIZE_P4, BIN_CENTERS, GRAD_CLIP_P4, LAMBDA_PHYSICS,
    LEARNING_RATE_P4, LR_FACTOR_P4, LR_PATIENCE_P4,
    MODELS_DIR, NUM_EPOCHS_P4, PINN_CKPT, PINN_HISTORY,
    PSI0_DIM, SIGMA_AI_DIM, WEIGHT_DECAY_P4,
)
from phase4.utils.logging import get_logger

logger = get_logger(__name__)
_BC = None   # bin centres tensor, lazily initialised


def _bin_centers(device):
    global _BC
    if _BC is None or _BC.device != device:
        import numpy as np
        _BC = torch.tensor(BIN_CENTERS, dtype=torch.float32, device=device)
    return _BC


class PINNTrainer:
    def __init__(self, model: BornRulePINN, device="cpu"):
        self.model  = model.to(device)
        self.device = device
        self.opt    = torch.optim.AdamW(model.parameters(),
                                        lr=LEARNING_RATE_P4, weight_decay=WEIGHT_DECAY_P4)
        self.sched  = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.opt, mode="min", factor=LR_FACTOR_P4, patience=LR_PATIENCE_P4)
        self.history: Dict = {k:[] for k in
            ["train_loss","train_ce","train_phys","val_loss","val_ce","val_acc"]}

    def _run_epoch(self, loader, train=True):
        self.model.train(train)
        totals = [0.]*3; n = 0; correct = total = 0

        ctx = torch.enable_grad() if train else torch.no_grad()
        with ctx:
            for psi0, sigma, alpha, y_bins, y_cont in loader:
                psi0  = psi0.to(self.device)
                sigma = sigma.to(self.device)
                alpha = alpha.to(self.device)
                y_bins = y_bins.to(self.device)
                y_cont = y_cont.to(self.device)

                probs, E, V = self.model(psi0, sigma, alpha)

                # Cross-entropy loss (classification into bins)
                ce_loss = F.cross_entropy(torch.log(probs + 1e-8), y_bins)

                # Physics loss: E[P] should match actual return direction
                phys = F.mse_loss(E, y_cont)

                loss = ce_loss + LAMBDA_PHYSICS * phys

                if train:
                    self.opt.zero_grad(); loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), GRAD_CLIP_P4)
                    self.opt.step()

                totals[0] += loss.item(); totals[1] += ce_loss.item()
                totals[2] += phys.item(); n += 1

                if not train:
                    pred_dir   = (E > 0)
                    true_dir   = (y_cont > 0)
                    correct   += (pred_dir == true_dir).sum().item()
                    total     += len(y_cont)

        avg = [t/n for t in totals]
        acc = correct/total if not train else 0.0
        return avg[0], avg[1], avg[2], acc

    def train(self, train_loader, val_loader,
              num_epochs=NUM_EPOCHS_P4, ckpt_path=PINN_CKPT):
        ckpt_path = Path(ckpt_path)
        ckpt_path.parent.mkdir(parents=True, exist_ok=True)
        best_loss = float("inf")

        for ep in range(1, num_epochs+1):
            tl,tc,tp,_    = self._run_epoch(train_loader, True)
            vl,vc,_,acc   = self._run_epoch(val_loader,   False)
            self.sched.step(vl)

            logger.info(f"Ep {ep:3d}/{num_epochs} | "
                        f"train loss={tl:.5f} ce={tc:.5f} phys={tp:.5f} | "
                        f"val loss={vl:.5f} ce={vc:.5f} acc={acc:.4f}")

            for k,v in zip(self.history, [tl,tc,tp,vl,vc,acc]):
                self.history[k].append(v)

            if vl < best_loss:
                best_loss = vl
                torch.save({"epoch":ep,
                            "model_state_dict": self.model.state_dict(),
                            "optimizer_state_dict": self.opt.state_dict(),
                            "val_loss": vl, "history": self.history,
                            "config": {"psi0_dim": self.model.psi0_proj[0].in_features,
                                       "sigma_ai_dim": PSI0_DIM,
                                       "num_bins": self.model.num_bins}},
                           ckpt_path)
                logger.info(f"  ✓ Best saved (loss={vl:.5f})")

        with open(PINN_HISTORY,"w") as f: json.dump(self.history, f, indent=2)
        return self.history

    @staticmethod
    def load_checkpoint(ckpt_path=PINN_CKPT, device="cpu"):
        ckpt  = torch.load(ckpt_path, map_location=device, weights_only=False)
        model = BornRulePINN()
        model.load_state_dict(ckpt["model_state_dict"])
        model.to(device)
        t = PINNTrainer(model, device)
        t.history = ckpt.get("history", t.history)
        logger.info(f"Loaded PINN ckpt (ep={ckpt['epoch']}, val_loss={ckpt['val_loss']:.5f})")
        return t, model
