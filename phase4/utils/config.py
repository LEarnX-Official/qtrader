"""Phase 4 configuration — Born Rule PINN for crypto collapse probability."""

from pathlib import Path
import numpy as np

from phase1.utils.config import Y_NPY, META_CSV
from phase2.utils.config import LATENTS_ALL, LATENTS_TRAIN, LATENTS_VAL, TRAIN_END, VAL_END
from phase3.utils.config import (
    SIGMA_AI_NPY, ALPHA_NPY,
    SIGMA_AI_TRAIN, SIGMA_AI_VAL,
    ALPHA_TRAIN, ALPHA_VAL,
)

ROOT         = Path(__file__).resolve().parents[2]
LOGS_P4      = ROOT / "logs"  / "phase4"
MODELS_DIR   = ROOT / "data"  / "models"
DATA_P4      = ROOT / "data"  / "processed" / "phase4"

PINN_CKPT      = MODELS_DIR / "pinn_collapse_best.pt"
PINN_HISTORY   = MODELS_DIR / "pinn_train_history.json"
COLLAPSE_PROBS = DATA_P4 / "collapse_probs.npy"
COLLAPSE_TRAIN = DATA_P4 / "collapse_probs_train.npy"
COLLAPSE_VAL   = DATA_P4 / "collapse_probs_val.npy"
EXPECTED_RETS  = DATA_P4 / "expected_returns.npy"
RETURN_VAR     = DATA_P4 / "return_variance.npy"

# Crypto returns are in 4h horizon log-returns.
# Range ±3% covers ~95% of hourly moves. Bin at 4h slightly wider → ±5%.
NUM_BINS    = 20
BIN_LOW     = -0.05
BIN_HIGH    =  0.05
BIN_EDGES   = np.linspace(BIN_LOW, BIN_HIGH, NUM_BINS + 1)
BIN_CENTERS = (BIN_EDGES[:-1] + BIN_EDGES[1:]) / 2

# Model dims
PSI0_DIM     = 32
SIGMA_AI_DIM = 512
ALPHA_DIM    = 1
DROPOUT      = 0.2

# Training
BATCH_SIZE_P4    = 256
NUM_EPOCHS_P4    = 100
LEARNING_RATE_P4 = 1e-4
WEIGHT_DECAY_P4  = 1e-4
GRAD_CLIP_P4     = 1.0
LAMBDA_PHYSICS   = 0.1
LR_FACTOR_P4     = 0.5
LR_PATIENCE_P4   = 5
NUM_WORKERS_P4   = 2

MIN_DIRECTION_ACC_P4 = 0.52
