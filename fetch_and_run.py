#!/usr/bin/env python3
"""
fetch_and_run.py — Live data download + inference for qtrader.

Self-contained runner for the LIVE trading data path:

    1. Download fresh 1h OHLCV for all tokens     (Binance / CMC, saved to data/raw/)
    2. Download supplementary data                 (fear/greed, funding, dominance, …)
    3. Load the freshly-saved data                 (load_live_data)
    4. Run phases 1-5 inference                     → target portfolio weights

Use this before a trading cycle to make sure the agent is acting on current
data, or as a standalone "what would the agent do right now?" check.

Usage
-----
    python fetch_and_run.py                 # download + infer, print weights
    python fetch_and_run.py --hours 1000    # how many hours of OHLCV to pull
    python fetch_and_run.py --no-fetch       # skip download, infer on cached CSVs
    python fetch_and_run.py --skip-supp      # OHLCV only, reuse cached supplementary
"""

import sys
import argparse
import datetime
from pathlib import Path

import numpy as np

BASE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE_DIR))

UTC = datetime.timezone.utc


def banner(msg: str) -> None:
    print(f"\n{'=' * 60}\n  {msg}\n{'=' * 60}")


def main() -> int:
    parser = argparse.ArgumentParser(description="qtrader live data fetch + inference")
    parser.add_argument("--hours", type=int, default=1000,
                        help="Hours of OHLCV history to fetch (default 1000)")
    parser.add_argument("--no-fetch", action="store_true",
                        help="Skip downloading; run on cached CSVs in data/raw/")
    parser.add_argument("--skip-supp", action="store_true",
                        help="Fetch OHLCV but reuse cached supplementary data")
    args = parser.parse_args()

    now = datetime.datetime.now(UTC)
    print("=" * 60)
    print(f"  qtrader — fetch & run   {now.strftime('%Y-%m-%d %H:%M UTC')}")
    print("=" * 60)

    from config import ALL_TOKENS, TRADE_TOKENS, RAW_DIR, SUPP_DIR

    # ── Step 1+2: Download fresh data ─────────────────────────────────────────
    if not args.no_fetch:
        from data.cmc_hub import CMCAgentHub
        hub = CMCAgentHub()

        banner(f"STEP 1 — Download OHLCV ({args.hours}h) for {len(ALL_TOKENS)} tokens")
        ohlcv_saved = hub.fetch_all_ohlcv_and_save(hours=args.hours)
        print(f"  Saved OHLCV for {len(ohlcv_saved)}/{len(ALL_TOKENS)} tokens → {RAW_DIR}")

        if not args.skip_supp:
            banner("STEP 2 — Download supplementary data (fear/greed, funding, …)")
            skills = hub.run_skills()
            hub.save_supplementary(skills)
            fg = skills.get("fear_greed", {})
            print(f"  Fear & Greed: {fg.get('value', '?')} ({fg.get('classification', '?')})")
            print(f"  x402 paid:    ${skills.get('x402_spent', 0):.4f} "
                  f"({skills.get('x402_calls', 0)} calls)")
        else:
            print("\n[skip] supplementary download skipped (--skip-supp)")
    else:
        print("\n[skip] download skipped (--no-fetch) — using cached CSVs")

    # ── Sanity: required CSVs present ─────────────────────────────────────────
    missing = [t for t in ALL_TOKENS
               if not (RAW_DIR / f"{t}USDT_1h_live.csv").exists()]
    if missing:
        print(f"\n[ERROR] Missing OHLCV CSVs for: {missing}")
        print("        Run without --no-fetch to download them first.")
        return 1

    # ── Step 3: Load fresh data ───────────────────────────────────────────────
    banner("STEP 3 — Load saved data")
    from inference.engine import load_live_data
    ohlcv, supp = load_live_data()
    n_btc = len(ohlcv["BTC"]) if "BTC" in ohlcv else 0
    print(f"  Loaded {len(ohlcv)} tokens | BTC history: {n_btc} candles")

    # ── Step 4: Run inference (phases 1-5) ────────────────────────────────────
    banner("STEP 4 — Inference (phases 1-5 → portfolio weights)")
    from inference.engine import QuantumTraderEngine
    engine = QuantumTraderEngine()

    result = engine.infer(
        ohlcv           = ohlcv,
        supp            = supp,
        current_weights = np.zeros(len(TRADE_TOKENS), dtype=np.float32),
        current_capital = 100.0,
        peak_capital    = 100.0,
        port_returns    = [],
    )

    weights = result["weights"]
    cash_w  = result["cash_weight"]

    banner("RESULT — Target Allocation")
    print(f"  {'CASH':6} {cash_w:6.1%}")
    for tok, w in sorted(zip(TRADE_TOKENS, weights), key=lambda x: -x[1]):
        bar = "█" * int(w * 40)
        print(f"  {tok:6} {w:6.1%}  {bar}")
    print(f"\n  (sum = {float(weights.sum()) + cash_w:.3f})")
    print("\nDone. To execute these weights on-chain, run:  python agent.py")
    return 0


if __name__ == "__main__":
    sys.exit(main())
