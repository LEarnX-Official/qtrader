"""
Central configuration for Quantum Trader Phase 1.
All paths, tokens, and hyperparameters live here.
"""

from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT          = Path(__file__).resolve().parents[2]   # qtrader/
DATA_DIR      = ROOT / "data"
RAW_DIR       = DATA_DIR / "raw"
SUPP_DIR      = DATA_DIR / "supplementary"
PROCESSED_DIR = DATA_DIR / "processed" / "phase1"
LOGS_DIR      = ROOT / "logs" / "phase1"

# Output artefacts
X_NPY    = PROCESSED_DIR / "X_sequences.npy"
Y_NPY    = PROCESSED_DIR / "y_targets.npy"
META_CSV = PROCESSED_DIR / "metadata.csv"

# ---------------------------------------------------------------------------
# Universe
# ---------------------------------------------------------------------------
TOKENS = ["BTC", "BNB", "SOL", "ETH", "XRP", "INJ", "DOGE", "LTC"]

# ---------------------------------------------------------------------------
# Data range
# ---------------------------------------------------------------------------
START_DATE = "2023-01-01"
END_DATE   = "2025-12-31"

# ---------------------------------------------------------------------------
# Feature engineering
# ---------------------------------------------------------------------------
# 168 hourly bars = 1 week look-back window
SEQUENCE_LENGTH = 168

# Predict return 4 hours ahead (less noisy than 1h, richer than 1d)
FORECAST_HORIZON = 4

# Rolling z-score window for price-derived features (720h = 30 days)
NORM_WINDOW = 720

# Minimum candles a token must have after cleaning
MIN_CANDLES = SEQUENCE_LENGTH + FORECAST_HORIZON + 1

# ---------------------------------------------------------------------------
# Volatility / indicator windows  (hours)
# ---------------------------------------------------------------------------
VOL_SHORT  = 168    # 1 week
VOL_LONG   = 720    # 1 month
MA_SHORT   = 168    # 1 week
MA_LONG    = 720    # 1 month
RSI_PERIOD = 14
ATR_PERIOD = 14
ADX_PERIOD = 14
BB_PERIOD  = 168
STOCH_K    = 14
STOCH_D    = 3
MACD_FAST  = 12
MACD_SLOW  = 26
MACD_SIG   = 9

# ---------------------------------------------------------------------------
# Cross-token correlation window (hours)
# ---------------------------------------------------------------------------
CORR_WINDOW = 24    # 24h rolling correlation with BTC
