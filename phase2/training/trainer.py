"""VAETrainer — identical logic to crypto/phase2, uses local imports."""

import json
from pathlib import Path
from typing import Dict, Tuple

import torch
from torch.utils.data import DataLoader

from phase2.models.vae import MarketVAE, vae_loss
from phase2.utils.config import (
    BETA_MAX, BETA_START, BETA_WARMUP, GRAD_CLIP,
    LEARNING_RATE, LR_FACTOR, LR_PATIENCE,
    MODELS_DIR, NUM_EPOCHS, TRAIN_HISTORY, VAE_CKPT, WEIGHT_DECAY,
)
from phase2.utils.logging import get_logger

logger = get_logger(__name__)


class VAETrainer:
    def __init__(self, model: MarketVAE, device="cpu", lr=LEARNING_RATE):
        self.model  = model.to(device)
        self.device = device
        self.opt    = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=WEIGHT_DECAY)
        self.sched  = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.opt, mode="min", factor=LR_FACTOR, patience=LR_PATIENCE)
        self.history = {k: [] for k in
                        ["train_loss","train_recon","train_kl","val_loss","val_recon","val_kl"]}
        self._iter  = 0

    def _beta(self):
        if self._iter >= BETA_WARMUP: return BETA_MAX
        return BETA_START + (BETA_MAX - BETA_START) * self._iter / BETA_WARMUP

    def _run_epoch(self, loader, train=True):
        self.model.train(train)
        totals = [0., 0., 0.]; n = 0
        ctx = torch.enable_grad() if train else torch.no_grad()
        with ctx:
            for batch in loader:
                x = (batch[0] if isinstance(batch, (list,tuple)) else batch).to(self.device)
                out  = self.model(x)
                loss = vae_loss(x, out, beta=self._beta() if train else BETA_MAX)
                if train:
                    self.opt.zero_grad()
                    loss.total.backward()
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), GRAD_CLIP)
                    self.opt.step()
                    self._iter += 1
                totals[0] += loss.total.item()
                totals[1] += loss.recon.item()
                totals[2] += loss.kl.item()
                n += 1
        return [t/n for t in totals]

    def train(self, train_loader, val_loader, num_epochs=NUM_EPOCHS, ckpt_path=VAE_CKPT):
        ckpt_path = Path(ckpt_path)
        ckpt_path.parent.mkdir(parents=True, exist_ok=True)
        best = float("inf")

        for ep in range(1, num_epochs+1):
            tl, tr, tk = self._run_epoch(train_loader, train=True)
            vl, vr, vk = self._run_epoch(val_loader,   train=False)
            self.sched.step(vl)

            logger.info(f"Ep {ep:3d}/{num_epochs} | "
                        f"train loss={tl:.5f} recon={tr:.5f} kl={tk:.5f} | "
                        f"val loss={vl:.5f} recon={vr:.5f} kl={vk:.5f} | "
                        f"lr={self.opt.param_groups[0]['lr']:.2e}")

            for k, v in zip(self.history, [tl,tr,tk,vl,vr,vk]):
                self.history[k].append(v)

            if vl < best:
                best = vl
                torch.save({"epoch": ep, "model_state_dict": self.model.state_dict(),
                            "optimizer_state_dict": self.opt.state_dict(),
                            "val_loss": vl, "history": self.history,
                            "config": {"seq_len": self.model.encoder.seq_len,
                                       "n_features": self.model.encoder.n_features,
                                       "latent_dim": self.model.latent_dim}},
                           ckpt_path)
                logger.info(f"  ✓ Best saved (val_loss={vl:.5f})")

        TRAIN_HISTORY.parent.mkdir(parents=True, exist_ok=True)
        with open(TRAIN_HISTORY, "w") as f: json.dump(self.history, f, indent=2)
        return self.history

    @staticmethod
    def load_checkpoint(ckpt_path=VAE_CKPT, device="cpu"):
        ckpt  = torch.load(ckpt_path, map_location=device, weights_only=False)
        cfg   = ckpt["config"]
        model = MarketVAE(cfg["seq_len"], cfg["n_features"], cfg["latent_dim"])
        model.load_state_dict(ckpt["model_state_dict"])
        model.to(device)
        logger.info(f"Loaded VAE ckpt (ep={ckpt['epoch']}, val_loss={ckpt['val_loss']:.5f})")
        t = VAETrainer(model, device)
        t.history = ckpt.get("history", t.history)
        return t, model
