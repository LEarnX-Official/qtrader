"""Phase 5 configuration — PPO portfolio optimisation.

Trading universe : 7 tokens (BTC excluded from portfolio/action)
State universe   : 8 tokens (BTC INCLUDED as context signal)

BTC is the dominant market driver — its Ψ₀, ΣAᵢ, and collapse probs
give the agent direct visibility of BTC's quantum state, which predicts
alt-coin behaviour far better than indirect signals (btc_corr, btc_dominance)
alone. The agent sees BTC but cannot hold it.
"""

from pathlib import Path
import numpy as np

ROOT       = Path(__file__).resolve().parents[2]
DATA_P5    = ROOT / "data" / "processed" / "phase5"
MODELS_DIR = ROOT / "data" / "models"
LOGS_DIR   = ROOT / "logs"  / "phase5"

PPO_CKPT    = MODELS_DIR / "ppo_agent_best.pt"
PPO_HISTORY = LOGS_DIR   / "ppo_history.json"

DATA_P5.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)
LOGS_DIR.mkdir(parents=True, exist_ok=True)

# ── Universe ──────────────────────────────────────────────────────────────────
ALL_TOKENS   = ["BTC", "BNB", "SOL", "ETH", "XRP", "INJ", "DOGE", "LTC"]
TRADE_TOKENS = ["BNB", "SOL", "ETH", "XRP", "INJ", "DOGE", "LTC"]   # 7 — action space
N_ASSETS     = len(TRADE_TOKENS)    # 7 — action / portfolio dimension
N_STATE_TOKENS = len(ALL_TOKENS)    # 8 — state observation includes BTC
BTC_IDX      = ALL_TOKENS.index("BTC")   # 0

# ── Date splits ───────────────────────────────────────────────────────────────
TRAIN_END  = "2025-06-30"
VAL_END    = "2025-12-31"

# ── State dimensions ──────────────────────────────────────────────────────────
PSI0_DIM      = 32
SIGMA_AI_DIM  = 512
ALPHA_DIM     = 1
PROB_DIM      = 20
PER_ASSET_DIM = PSI0_DIM + SIGMA_AI_DIM + ALPHA_DIM + PROB_DIM   # 565
PORTFOLIO_METRICS = 3    # pnl, vol, drawdown

# State = 8 tokens × 565 (agent observes BTC too) + 7 portfolio weights + 3 metrics
STATE_DIM  = PER_ASSET_DIM * N_STATE_TOKENS + N_ASSETS + PORTFOLIO_METRICS
#          = 565 × 8 + 7 + 3 = 4530

# ── PINN bin centres (must match Phase 4) ─────────────────────────────────────
BIN_EDGES   = np.linspace(-0.05, 0.05, PROB_DIM + 1).tolist()
BIN_CENTERS = [(BIN_EDGES[i] + BIN_EDGES[i+1]) / 2 for i in range(PROB_DIM)]

# ── Environment ───────────────────────────────────────────────────────────────
INITIAL_CAPITAL   = 100_000.0
TRANSACTION_COST  = 0.001      # 10 bps
SLIPPAGE          = 0.0005     # 5 bps
MAX_POSITION      = 0.40       # 40% per asset max
EPISODE_LEN_TRAIN = 168 * 3    # 504 steps (~3 days) — shorter = more regime diversity
EPISODE_LEN_VAL   = None       # full val period

# ── Reward ────────────────────────────────────────────────────────────────────
# Rewards are 4h log-returns ~ N(0, 0.008).
# Keep all penalty terms at the same scale to avoid dominance.
LAMBDA_RISK      = 0.1    # vol² penalty
LAMBDA_TURNOVER  = 0.001  # turnover penalty
LAMBDA_DRAWDOWN  = 5.0    # drawdown penalty — fires earlier and harder
LAMBDA_CALIB     = 0.01   # calibration bonus
LAMBDA_RETURN    = 0.2    # cumulative return bonus — prevents indifference to abs losses
MAX_DRAWDOWN_THR = 0.10   # penalise from 10% drawdown onward

# ── PPO hyperparameters ───────────────────────────────────────────────────────
LR_ACTOR     = 3e-4
LR_CRITIC    = 1e-3
GAMMA        = 0.99
GAE_LAMBDA   = 0.95
CLIP_EPS     = 0.2
PPO_EPOCHS   = 4
BATCH_SIZE   = 512
GRAD_CLIP    = 0.5
NUM_EPISODES = 500
EVAL_EVERY   = 10

# ── Actor architecture ────────────────────────────────────────────────────────
ASSET_EMBED_DIM = 128
HIDDEN_DIM      = 256
NUM_HEADS       = 4
DROPOUT         = 0.1
KELLY_FRACTION  = 0.5
