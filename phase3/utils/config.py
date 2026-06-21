"""Phase 3 configuration — Observer Aggregator Transformer for crypto ΣAᵢ."""

from pathlib import Path
from phase1.utils.config import Y_NPY, META_CSV
from phase2.utils.config import LATENTS_ALL, LATENTS_TRAIN, LATENTS_VAL, TRAIN_END, VAL_END

ROOT         = Path(__file__).resolve().parents[2]
LOGS_P3      = ROOT / "logs"  / "phase3"
MODELS_DIR   = ROOT / "data"  / "models"
DATA_P3      = ROOT / "data"  / "processed" / "phase3"

SIGMA_AI_NPY      = DATA_P3 / "sigma_ai.npy"          # (N, 512)
ALPHA_NPY         = DATA_P3 / "alignment_scores.npy"  # (N, 1)
SIGMA_AI_TRAIN    = DATA_P3 / "sigma_ai_train.npy"
SIGMA_AI_VAL      = DATA_P3 / "sigma_ai_val.npy"
ALPHA_TRAIN       = DATA_P3 / "alpha_train.npy"
ALPHA_VAL         = DATA_P3 / "alpha_val.npy"

OBS_CKPT     = MODELS_DIR / "observer_transformer_best.pt"
OBS_HISTORY  = MODELS_DIR / "observer_train_history.json"

VIZ_OBS_LOSS  = LOGS_P3 / "observer_loss.png"
VIZ_OBS_ALIGN = LOGS_P3 / "alignment_analysis.png"
VIZ_OBS_SIGMA = LOGS_P3 / "sigma_ai_norms.png"

# ── Crypto modality layout ─────────────────────────────────────────────────
# Total = 256, sourced from supplementary data aligned to each sequence step.
#
#   psi0_latent   : 32   — VAE latent vector for this sequence
#   funding       : 16   — funding_rate, funding_abs, funding_momentum × repeated
#   fear_greed    : 16   — fear_greed_value, fear_greed_class × repeated
#   onchain       : 32   — vol_mcap_ratio, onchain_pct_chg, onchain_trades, own_mcap
#   dominance     : 32   — btc_dominance_pct, total_mcap_8tokens
#   market_tech   : 96   — price/volume/technical features summary (rolling stats)
#   macro_corr    : 32   — btc_corr, google_trend, cross-token context
# ──────────────────────────────────────────────────────────────────────────
MODALITY_DIMS = {
    "psi0_latent":  32,
    "funding":      16,
    "fear_greed":   16,
    "onchain":      32,
    "dominance":    32,
    "market_tech":  96,
    "macro_corr":   32,
}
INPUT_DIM = sum(MODALITY_DIMS.values())   # 256
SIGMA_DIM = 512

# Model
D_MODEL    = 128
NUM_HEADS  = 8
NUM_LAYERS = 6
D_FF       = 512
DROPOUT    = 0.1

# Training
BATCH_SIZE_P3    = 256
NUM_EPOCHS_P3    = 50
LEARNING_RATE_P3 = 3e-4
WEIGHT_DECAY_P3  = 1e-4
GRAD_CLIP_P3     = 1.0
ALIGN_LOSS_WEIGHT = 0.1
NUM_WORKERS_P3   = 2

MIN_DIRECTION_ACC = 0.52
