"""Post-training analysis: extract latents, PCA viz, reconstruction check."""

from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader

from phase2.models.vae import MarketVAE
from phase2.utils.config import LATENTS_ALL, VIZ_DIMS, VIZ_LATENT, VIZ_RECON, LOGS_P2
from phase2.utils.logging import get_logger

logger = get_logger(__name__)


class VAEAnalyzer:
    def __init__(self, model: MarketVAE, device="cpu"):
        self.model  = model.to(device).eval()
        self.device = device

    @torch.no_grad()
    def extract_latents(self, loader: DataLoader) -> np.ndarray:
        out = []
        for batch in loader:
            x = (batch[0] if isinstance(batch, (list,tuple)) else batch).to(self.device)
            out.append(self.model.encode(x).cpu().numpy())
        return np.concatenate(out)

    @torch.no_grad()
    def reconstruction_error(self, loader: DataLoader) -> float:
        errs = []
        for batch in loader:
            x = (batch[0] if isinstance(batch, (list,tuple)) else batch).to(self.device)
            out = self.model(x)
            errs.append(((x - out.x_recon)**2).mean(dim=[1,2]).cpu().numpy())
        return float(np.concatenate(errs).mean())

    def run_all(self, train_loader, val_loader, y_val, history) -> Tuple[Dict, np.ndarray]:
        LOGS_P2.mkdir(parents=True, exist_ok=True)

        logger.info("Extracting val latents …")
        latents_val = self.extract_latents(val_loader)

        recon_err = self.reconstruction_error(val_loader)
        logger.info(f"Val recon error: {recon_err:.5f}")

        # PCA
        from sklearn.decomposition import PCA
        pca = PCA(n_components=2)
        pca.fit(latents_val)
        pca_var = float(pca.explained_variance_ratio_.sum())
        logger.info(f"PCA 2-component explained variance: {pca_var:.1%}")

        # Active latent dims (std > 0.1)
        active = int((latents_val.std(axis=0) > 0.1).sum())
        logger.info(f"Active latent dims (std>0.1): {active}/{latents_val.shape[1]}")

        # KL from last history entry
        val_kl = history.get("val_kl", [0])[-1] if history else 0.0

        metrics = {
            "val_recon_loss":     recon_err,
            "val_kl":             val_kl,
            "pca_explained_var":  pca_var,
            "active_latent_dims": active,
        }

        self._plot_latents(latents_val, y_val)
        self._plot_dims(latents_val)
        self._plot_loss(history)

        return metrics, latents_val

    def _plot_latents(self, latents, y_val):
        try:
            import matplotlib; matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            from sklearn.decomposition import PCA
            z2d = PCA(n_components=2).fit_transform(latents[:5000])
            plt.figure(figsize=(10,7))
            sc = plt.scatter(z2d[:,0], z2d[:,1], c=y_val[:5000], cmap="RdYlGn",
                             alpha=0.5, s=8)
            plt.colorbar(sc, label="4h return")
            plt.title("Latent Space PCA (val set)")
            plt.savefig(VIZ_LATENT, dpi=120, bbox_inches="tight")
            plt.close()
            logger.info(f"Saved {VIZ_LATENT}")
        except Exception as e:
            logger.warning(f"Plot failed: {e}")

    def _plot_dims(self, latents):
        try:
            import matplotlib; matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            d = latents.shape[1]
            fig, axes = plt.subplots(4, d//4, figsize=(16,12))
            for i, ax in enumerate(axes.flatten()):
                ax.hist(latents[:,i], bins=40, alpha=0.7)
                ax.axvline(0, color="red", lw=0.8)
                ax.set_title(f"z{i} σ={latents[:,i].std():.2f}")
            plt.tight_layout()
            plt.savefig(VIZ_DIMS, dpi=120, bbox_inches="tight")
            plt.close()
            logger.info(f"Saved {VIZ_DIMS}")
        except Exception as e:
            logger.warning(f"Dim plot failed: {e}")

    def _plot_loss(self, history):
        if not history: return
        try:
            import matplotlib; matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            from phase2.utils.config import VIZ_LOSS
            fig, axes = plt.subplots(1,3, figsize=(15,4))
            for ax, key in zip(axes, ["loss","recon","kl"]):
                ax.plot(history.get(f"train_{key}",[]), label="train")
                ax.plot(history.get(f"val_{key}",[]),   label="val")
                ax.set_title(key); ax.legend()
            plt.tight_layout()
            plt.savefig(VIZ_LOSS, dpi=120, bbox_inches="tight")
            plt.close()
        except Exception as e:
            logger.warning(f"Loss plot failed: {e}")
