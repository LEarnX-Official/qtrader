"""Phase 2 configuration — VAE for Ψ₀ state estimation (crypto hourly data)."""

from pathlib import Path

from phase1.utils.config import (
    PROCESSED_DIR as DATA_P1,
    X_NPY, Y_NPY, META_CSV,
    SEQUENCE_LENGTH, TOKENS,
)

ROOT       = Path(__file__).resolve().parents[2]
MODELS_DIR = ROOT / "data" / "models"
LOGS_P2    = ROOT / "logs" / "phase2"
DATA_P2    = ROOT / "data" / "processed" / "phase2"

VAE_CKPT      = MODELS_DIR / "vae_best.pt"
LATENTS_ALL   = DATA_P2 / "latent_representations.npy"
LATENTS_TRAIN = DATA_P2 / "latent_train.npy"
LATENTS_VAL   = DATA_P2 / "latent_val.npy"
TRAIN_HISTORY = MODELS_DIR / "vae_train_history.json"

VIZ_LATENT = LOGS_P2 / "latent_space_viz.png"
VIZ_DIMS   = LOGS_P2 / "latent_dimensions.png"
VIZ_RECON  = LOGS_P2 / "reconstructions.png"
VIZ_LOSS   = LOGS_P2 / "training_loss.png"

# Model
SEQ_LEN    = SEQUENCE_LENGTH   # 168 hours
N_FEATURES = 40                # from Phase 1
LATENT_DIM = 32

# Training
BATCH_SIZE    = 256
NUM_EPOCHS    = 100
LEARNING_RATE = 1e-4
WEIGHT_DECAY  = 1e-5
NUM_WORKERS   = 2
GRAD_CLIP     = 1.0

# β-VAE annealing
BETA_START  = 0.01
BETA_MAX    = 0.1
BETA_WARMUP = 10_000

# LR scheduler
# patience=10 gives the val loss more room to plateau before reducing LR.
# With 100 epochs, patience=5 was too aggressive — fired at ep 28 and
# exhausted all LR budget by ep 65 with no benefit.
LR_PATIENCE = 10
LR_FACTOR   = 0.5

# Date splits
# Train : 2023-01-01 → 2025-06-30  (2.5 years)
# Val   : 2025-07-01 → 2025-12-31  (6 months)
TRAIN_END = "2025-06-30"
VAL_END   = "2025-12-31"

# Recon threshold calibrated for hourly crypto data (z-score normalised, 40 features).
# Daily NASDAQ used 0.5 — hourly crypto is much noisier, 3.0 is realistic.
RECON_THRESHOLD = 3.0

# Minimum active latent dims (std > 0.1).
# Crypto across 8 tokens has fewer independent factors than 100 NASDAQ stocks.
# 8+ active dims is healthy for this dataset.
MIN_ACTIVE_DIMS = 8
