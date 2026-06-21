"""
BNB Hack: Quantum Trader — Central Configuration
All keys loaded from .env, all constants defined here.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# Load .env from project folder
ENV_PATH = Path(__file__).resolve().parent / ".env"
load_dotenv(ENV_PATH)

# ── API Keys ──────────────────────────────────────────────────────────────────
CMC_API_KEY          = os.getenv("CMC_API_KEY", "")
TWAK_API_KEY         = os.getenv("TWAK_API_KEY", "")
AGENT_WALLET_ADDRESS = os.getenv("AGENT_WALLET_ADDRESS", "")
TWAK_WALLET_ADDRESS  = os.getenv("TWAK_WALLET_ADDRESS", "")
WALLET_PASSWORD      = os.getenv("WALLET_PASSWORD", "")
ASTER_API_KEY        = os.getenv("ASTER_API_KEY", "")
ASTER_API_SECRET     = os.getenv("ASTER_API_SECRET", "")
ASTER_USER_ADDRESS   = os.getenv("ASTER_USER_ADDRESS", "")

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR        = Path(__file__).resolve().parent
QT_DIR          = BASE_DIR.parent / "quantum_trader"   # main architecture (model class defs)
MODELS_DIR      = BASE_DIR / "data" / "models"         # trained weights bundled in qtrader
DATA_DIR        = BASE_DIR / "data"
RAW_DIR         = DATA_DIR / "raw"
SUPP_DIR        = DATA_DIR / "supplementary"
PROC_DIR        = DATA_DIR / "processed"
LOGS_DIR        = DATA_DIR / "logs"
RESULTS_DIR     = BASE_DIR / "results"

for d in [RAW_DIR, SUPP_DIR, PROC_DIR, LOGS_DIR, RESULTS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# ── Universe ──────────────────────────────────────────────────────────────────
ALL_TOKENS   = ["BTC", "BNB", "SOL", "ETH", "XRP", "INJ", "DOGE", "LTC"]
TRADE_TOKENS = ["BNB", "SOL", "ETH", "XRP", "INJ", "DOGE", "LTC"]

# Of the 7 trained tokens, only these are on the official competition
# eligible-token list. BNB and SOL are NOT eligible — trades in them
# do not count. The agent zeroes BNB/SOL weights and redistributes to these.
ELIGIBLE_TRADE_TOKENS = ["ETH", "XRP", "INJ", "DOGE", "LTC"]
INELIGIBLE_TOKENS     = ["BNB", "SOL"]

# BSC BEP-20 contract addresses for eligible competition tokens
# Used by TWAK for on-chain swaps
TOKEN_ADDRESSES = {
    "BNB":  "0xbb4CdB9CBd36B01bD1cBaEBF2De08d9173bc095c",  # WBNB
    "ETH":  "0x2170Ed0880ac9A755fd29B2688956BD959F933F8",  # ETH BEP-20
    "XRP":  "0x1D2F0da169ceB9fC7B3144628dB156f3F6c60dBE",  # XRP BEP-20
    "DOGE": "0xbA2aE424d960c26247Dd6c32edC70B295c744C43",  # DOGE BEP-20
    "LTC":  "0x4338665CBB7B2485A8855A139b75D5e34AB0DB94",  # LTC BEP-20
    "INJ":  "0xa2B726B1145A4773F68593CF171187d8EBe4d495",  # INJ BEP-20
    "SOL":  "0x570A5D26f7765Ecb712C0924E4De545B89fD43dF",  # SOL BEP-20
    "USDT": "0x55d398326f99059fF775485246999027B3197955",  # USDT BEP-20
}
USDT_ADDRESS = TOKEN_ADDRESSES["USDT"]

# ── Trading Parameters ────────────────────────────────────────────────────────
INITIAL_CAPITAL      = 100.0       # USD (paper trade) / real USDT (live)
TRANSACTION_COST     = 0.001       # 10 bps
SLIPPAGE             = 0.0005      # 5 bps
MAX_POSITION         = 0.40        # 40% max per asset
MIN_TRADE_THRESHOLD  = 0.01        # ignore rebalances < 1% of capital (avoid dust fees)
CYCLE_HOURS          = 1           # full inference every 1h
CHECK_INTERVAL_MIN   = 15          # price check + profit manager every 15min

# ── Risk Management ───────────────────────────────────────────────────────────
DD_WARN_THRESHOLD    = 0.10        # 10% drawdown → alert + reduce sizes 50%
DD_STOP_THRESHOLD    = 0.15        # 15% drawdown → hard stop, go 100% cash
DD_COMPETITION_CAP   = 0.30        # 30% → disqualified by competition rules
MIN_TRADES_PER_DAY   = 1           # competition requires 1 trade/day minimum
FORCE_TRADE_HOURS    = 23          # if no trade in 23h → force micro-rebalance

# ── Mode ──────────────────────────────────────────────────────────────────────
# Set DRY_RUN=True for paper trading (Jun 9-21)
# Set DRY_RUN=False for live trading (Jun 22+)
DRY_RUN = True

# ── Competition ───────────────────────────────────────────────────────────────
COMPETITION_CONTRACT = "0x212c61b9b72c95d95bf29cf032f5e5635629aed5"
COMPETITION_START    = "2026-06-22T00:00:00Z"
COMPETITION_END      = "2026-06-28T23:59:59Z"

# ── CMC API ───────────────────────────────────────────────────────────────────
CMC_BASE_URL         = "https://pro-api.coinmarketcap.com"
CMC_AGENT_HUB_URL    = "https://coinmarketcap.com/api/agent"
CMC_MCP_SERVER_URL   = "https://mcp.coinmarketcap.com"   # CMC MCP server
CMC_HEADERS          = {
    "X-CMC_PRO_API_KEY": CMC_API_KEY,
    "Accept":            "application/json",
}

# ── x402 Micropayment Protocol ────────────────────────────────────────────────
X402_ENABLED         = True       # Pay per CMC Agent Hub request via TWAK
X402_COST_PER_CALL   = 0.001      # $0.001 per data request
X402_COST_OHLCV      = 0.001      # per OHLCV fetch
X402_COST_QUOTES     = 0.001      # per quotes fetch
X402_COST_FEAR_GREED = 0.0005     # per fear/greed fetch
X402_COST_GLOBAL     = 0.0005     # per global metrics fetch

# ── BNB Agent SDK ─────────────────────────────────────────────────────────────
BNB_AGENT_NETWORK    = "bsc-mainnet"
BNB_AGENT_REGISTRY   = "0x8004A169FB4a3325136EB29fA0ceB6D2e539a432"
BNB_AGENT_PORT       = 8003
BNB_AGENT_NAME       = "quantum-trader-ppo"

# CMC coin IDs for our tokens
CMC_IDS = {
    "BTC":  1,
    "BNB":  1839,
    "SOL":  5426,
    "ETH":  1027,
    "XRP":  52,
    "INJ":  14328,
    "DOGE": 74,
    "LTC":  2,
}

# ── Binance (OHLCV fallback) ──────────────────────────────────────────────────
BINANCE_BASE_URL     = "https://api.binance.com/api/v3"
BINANCE_FUTURES_URL  = "https://fapi.binance.com/fapi/v1"

# ── Feature Engineering (must match training) ─────────────────────────────────
SEQUENCE_LENGTH      = 168         # 1 week of hourly bars
FORECAST_HORIZON     = 4           # 4h ahead
NORM_WINDOW          = 720         # 30 days rolling z-score

# ── TWAK / BSC ────────────────────────────────────────────────────────────────
BSC_RPC_URL          = "https://bsc-dataseed1.binance.org/"
BSC_CHAIN_ID         = 56
PANCAKESWAP_ROUTER   = "0x10ED43C718714eb63d5aA57B78B54704E256024E"

# ── Telegram Alerts ───────────────────────────────────────────────────────────
# Set these after creating a Telegram bot
TELEGRAM_BOT_TOKEN   = os.getenv("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_CHAT_ID     = os.getenv("TELEGRAM_CHAT_ID", "")


def validate():
    """Check all required keys are present."""
    missing = []
    required = {
        "CMC_API_KEY":          CMC_API_KEY,
        "TWAK_API_KEY":         TWAK_API_KEY,
        "AGENT_WALLET_ADDRESS": AGENT_WALLET_ADDRESS,
        "WALLET_PASSWORD":      WALLET_PASSWORD,
    }
    for name, val in required.items():
        if not val:
            missing.append(name)
    if missing:
        raise EnvironmentError(f"Missing required env vars: {missing}")
    print(f"Config OK — wallet={AGENT_WALLET_ADDRESS[:10]}...  dry_run={DRY_RUN}")


if __name__ == "__main__":
    validate()
    print(f"Models dir : {MODELS_DIR}")
    print(f"Data dir   : {DATA_DIR}")
    print(f"DRY_RUN    : {DRY_RUN}")
