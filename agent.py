"""
Quantum Trader — Main Agent Loop
BNB Hack: AI Trading Agent Edition ⚡ CoinMarketCap × Trust Wallet

Full pipeline every 4 hours:
  1. CMC Agent Hub  → live quotes, fear/greed, dominance (x402 paid)
  2. CMC Skills     → momentum, regime, sentiment signals
  3. Inference      → phases 1-4 → PPO weights
  4. Risk Manager   → guardrails check
  5. TWAK           → local signing + autonomous BSC execution
  6. BNB Agent SDK  → on-chain job settlement
  7. Telegram       → alerts

Usage:
  python agent.py                    # single cycle (for cron)
  python agent.py --loop             # continuous every 4h
  python agent.py --register         # register on-chain (before Jun 22)
  python agent.py --serve            # start BNB Agent strategy server
  python agent.py --status           # show all component status
  python agent.py --dry-run false    # override to live mode
"""

import sys
import time
import argparse
import datetime
import traceback
import numpy as np
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
# Phase packages (phase1-5) are bundled inside qtrader — fully standalone.
sys.path.insert(0, str(BASE_DIR))

from config import (
    DRY_RUN, CYCLE_HOURS, AGENT_WALLET_ADDRESS,
    INITIAL_CAPITAL, TRADE_TOKENS, ALL_TOKENS,
    BASE_CURRENCY,
)

UTC = datetime.timezone.utc


def run_cycle(
    engine,
    trader,
    risk,
    twak,
    cmc_hub,
    bnb_agent,
    profit,
    last_full_fetch: datetime.datetime,
) -> datetime.datetime:
    """
    One full 4h cycle: CMC data → inference → risk → TWAK execute → alert.
    """
    now = datetime.datetime.now(UTC)
    print(f"\n{'='*60}")
    print(f"CYCLE — {now.strftime('%Y-%m-%d %H:%M UTC')} | "
          f"{'PAPER' if DRY_RUN else '🔴 LIVE'}")
    print(f"{'='*60}")

    from alerts import telegram

    # ── Step 1: CMC Agent Hub — fetch live data (x402 paid) ──────────────────
    print("\n[1/7] CMC Agent Hub — fetching live data...")
    try:
        # x402 pays for each CMC data call
        skills_result = cmc_hub.run_skills()

        # Save supplementary data for Phase 1 feature engineering
        cmc_hub.save_supplementary(skills_result)

        # Fetch and save OHLCV
        hours_since_full = (now - last_full_fetch).total_seconds() / 3600
        if hours_since_full >= 4:
            cmc_hub.fetch_all_ohlcv_and_save(hours=1000)
            last_full_fetch = now

        fg      = skills_result["fear_greed"]
        metrics = skills_result["global"]
        print(f"  Fear&Greed: {fg['value']} ({fg['classification']})")
        print(f"  BTC Dom: {metrics.get('btc_dominance',0):.1f}%")
        print(f"  x402 paid: ${skills_result['x402_spent']:.4f} "
              f"({skills_result['x402_calls']} calls)")

    except Exception as e:
        print(f"  [Agent] CMC fetch failed: {e}")
        telegram.alert_error(f"CMC fetch failed: {e}")
        return last_full_fetch

    # ── Step 2: CMC Skills — print signals ───────────────────────────────────
    print("\n[2/7] CMC Skills — strategy signals...")
    print(f"  Regime: {skills_result['regime'].split(chr(10))[1]}")

    # ── Step 3: Load data + run inference ────────────────────────────────────
    print("\n[3/7] Inference — phases 1-4 → PPO weights...")
    try:
        from inference.engine import load_live_data
        ohlcv, supp = load_live_data()

        result = engine.infer(
            ohlcv           = ohlcv,
            supp            = supp,
            current_weights = trader.weights,
            current_capital = trader.capital,
            peak_capital    = trader.peak_capital,
            port_returns    = trader.port_returns,
        )
        weights = result["weights"]
        cash_w  = result["cash_weight"]

        # ── Minimum-weight filter ─────────────────────────────────────────────
        # Only hold a position if weight >= 5%. Smaller positions → cash.
        # Avoids dust trades that waste gas/fees; concentrates conviction.
        MIN_WEIGHT = 0.05
        freed = float(weights[weights < MIN_WEIGHT].sum())
        weights = np.where(weights < MIN_WEIGHT, 0.0, weights).astype(np.float32)
        cash_w  = min(1.0, cash_w + freed)
        if freed > 0:
            print(f"  [Filter] Zeroed positions <5% → +{freed:.1%} to cash")

    except Exception as e:
        print(f"  [Agent] Inference failed: {e}")
        telegram.alert_error(f"Inference failed: {e}")
        return last_full_fetch

    # ── Step 4: Risk Manager ──────────────────────────────────────────────────
    print("\n[4/7] Risk Manager — checking guardrails...")
    adj_weights, adj_cash, status = risk.check(
        capital = trader.capital,
        weights = weights,
        cash_w  = cash_w,
    )

    if status == "stop":
        telegram.alert_risk("stop", risk.current_drawdown(trader.capital), trader.capital)
    elif status == "warn":
        telegram.alert_risk("warn", risk.current_drawdown(trader.capital), trader.capital)

    print(f"  Status: {status} | DD: {risk.current_drawdown(trader.capital):.2%}")

    # ── Step 4b: Profit Manager — take-profit + trailing stop (runs FIRST) ────
    print(f"\n[4b/7] Profit Manager — checking take-profit / trailing stop...")
    prices = trader.get_current_prices()
    profit.update_prices(prices, trader.weights)
    adj_weights, adj_cash, tp_actions = profit.apply(adj_weights, prices, adj_cash)
    if tp_actions:
        for token, action in tp_actions.items():
            print(f"  {token}: {action}")

    # Show open position P&L
    pos_status = profit.get_status(prices)
    if pos_status:
        print(f"  Open positions P&L:")
        for tok, s in pos_status.items():
            print(f"    {tok}: entry=${s['entry']} now=${s['current']} "
                  f"pnl={s['pnl']} trail={s['trail']}")

    # ── Step 5: Decide whether to trade ──────────────────────────────────────
    execute, reason = risk.should_execute(
        old_weights = trader.weights,
        new_weights = adj_weights,
        old_cash    = trader.cash_weight,
        new_cash    = adj_cash,
        status      = status,
    )
    if tp_actions and not execute:
        execute = True
        reason  = "profit_manager"

    if not execute:
        # No trade this cycle — but still LOG the portfolio snapshot so the
        # monitor and PnL curve update every cycle (hold = a valid data point).
        prices_now = trader.get_current_prices()
        cap_now    = trader.portfolio_value(prices_now)
        trader.capital = cap_now
        if cap_now > trader.peak_capital:
            trader.peak_capital = cap_now
        trader._save_trade({
            "datetime":    now.isoformat(),
            "capital":     round(cap_now, 2),
            "port_return": 0.0,
            "cash_weight": round(trader.cash_weight, 4),
            "reason":      f"hold ({reason})",
            "dry_run":     DRY_RUN,
            "tx_hashes":   [],
            "n_trades":    0,
            "x402_paid":   round(skills_result.get("x402_spent", 0), 6),
            **{f"w_{t}": round(float(trader.weights[i]), 4)
               for i, t in enumerate(TRADE_TOKENS)},
        })
        print(f"\n[5-7/7] Holding (no trade): {reason} — snapshot logged")
        return last_full_fetch

    # ── Step 5: TWAK guardrail check per token ────────────────────────────────
    print(f"\n[5/7] TWAK — checking per-token guardrails...")
    prices = trader.get_current_prices()
    capital = trader.portfolio_value(prices)

    for i, token in enumerate(TRADE_TOKENS):
        w = float(adj_weights[i])
        if w > 0.01:
            allowed, gr_reason = twak.guardrails.check_trade(
                token=token, amount_usd=capital * w,
                capital=capital, peak_capital=trader.peak_capital)
            if not allowed:
                print(f"  {token} blocked: {gr_reason}")
                adj_weights[i] = 0.0

    # Re-normalize after any blocked tokens
    total = adj_weights.sum()
    if total > 0:
        adj_weights = adj_weights / total * (1.0 - adj_cash)

    # ── Step 6: Execute via TWAK ──────────────────────────────────────────────
    print(f"\n[6/7] TWAK — executing rebalance ({reason})...")
    try:
        target_holdings = trader._calc_target_holdings(
            adj_weights, adj_cash, capital, prices)
        trades = trader._calc_trades(trader.holdings, target_holdings, prices)

        tx_hashes      = []
        executed_count = 0
        for t in trades:
            if t["action"] == "buy":
                swap_result = twak.swap(
                    from_token=BASE_CURRENCY, to_token=t["token"],
                    amount_usd=t["amount_usd"],
                    capital=capital, peak_capital=trader.peak_capital)
            else:
                swap_result = twak.swap(
                    from_token=t["token"], to_token=BASE_CURRENCY,
                    amount_usd=t["amount_usd"],
                    capital=capital, peak_capital=trader.peak_capital)

            if swap_result.get("success") and swap_result.get("tx_hash"):
                executed_count += 1
                tx_hashes.append({
                    "token": t["token"],
                    "tx":    swap_result["tx_hash"],
                    "scan":  swap_result.get("bsc_scan", ""),
                })
                twak.guardrails.record_trade()

        # Only update portfolio state if at least one swap actually executed.
        # If everything was blocked (e.g. ineligible token), keep old state —
        # no phantom capital change, no false trade recorded.
        if executed_count == 0:
            print(f"  [Agent] No swaps executed (all blocked) — state unchanged")
            return last_full_fetch

        # Update trader state
        trader.weights     = adj_weights.copy()
        trader.cash_weight = adj_cash
        trader.holdings    = target_holdings
        risk.record_trade()

        # Update profit manager entry prices for new positions
        for i, token in enumerate(TRADE_TOKENS):
            profit.update_entry(token, prices.get(token, 0), float(adj_weights[i]))

        new_capital     = trader.portfolio_value(prices)
        trader.capital  = new_capital
        if new_capital > trader.peak_capital:
            trader.peak_capital = new_capital

        port_ret = (new_capital - capital) / max(capital, 1e-8)
        trader.port_returns.append(port_ret)
        total_return = (new_capital - INITIAL_CAPITAL) / INITIAL_CAPITAL

        # Save trade record
        trader._save_trade({
            "datetime":     now.isoformat(),
            "capital":      round(new_capital, 2),
            "port_return":  round(port_ret, 6),
            "cash_weight":  round(adj_cash, 4),
            "reason":       reason,
            "dry_run":      DRY_RUN,
            "tx_hashes":    tx_hashes,
            "n_trades":     len(trades),
            "x402_paid":    round(skills_result["x402_spent"], 6),
            **{f"w_{t}": round(float(adj_weights[i]), 4)
               for i, t in enumerate(TRADE_TOKENS)},
        })

        print(f"  Capital: ${new_capital:,.2f} | "
              f"Return: {total_return:+.2%} | "
              f"Trades: {len(trades)}")

    except Exception as e:
        print(f"  [Agent] Trade execution failed: {e}")
        traceback.print_exc()
        telegram.alert_error(f"Execution failed: {e}")
        return last_full_fetch

    # ── Step 7: Alerts ────────────────────────────────────────────────────────
    print(f"\n[7/7] Sending Telegram alert...")
    telegram.alert_trade(
        weights      = adj_weights,
        cash_w       = adj_cash,
        capital      = trader.capital,
        total_return = (trader.capital - INITIAL_CAPITAL) / INITIAL_CAPITAL,
        reason       = reason,
        tx_hashes    = tx_hashes,
    )

    # Daily summary
    if now.hour == 0 and now.minute < 15:
        from execution.trader import TWAKTrader
        summary = trader.get_portfolio_summary()
        telegram.alert_daily_summary(
            capital      = trader.capital,
            total_return = (trader.capital - INITIAL_CAPITAL) / INITIAL_CAPITAL,
            max_dd       = float(summary["max_drawdown"].strip("%")) / 100,
            n_trades     = risk.trades_today(),
            weights      = adj_weights,
            cash_w       = adj_cash,
        )

    return last_full_fetch


def show_status(engine, trader, twak, cmc_hub, bnb_agent, risk):
    """Print full status of all components."""
    print("\n" + "="*60)
    print("QUANTUM TRADER — Component Status")
    print("="*60)

    print("\n[TWAK]")
    for k, v in twak.get_status().items():
        print(f"  {k}: {v}")

    print("\n[BNB Agent]")
    for k, v in bnb_agent.get_status().items():
        print(f"  {k}: {v}")

    print("\n[CMC Hub]")
    print(f"  x402 calls this session: {len(cmc_hub.get_x402_log() if hasattr(cmc_hub, 'get_x402_log') else [])}")

    print("\n[Risk]")
    for k, v in risk.get_status_report(trader.capital).items():
        print(f"  {k}: {v}")

    print("\n[Trader]")
    summary = trader.get_portfolio_summary()
    for k, v in summary.items():
        if k not in ("holdings",):
            print(f"  {k}: {v}")

    print("="*60)


def main():
    parser = argparse.ArgumentParser(description="Quantum Trader Agent")
    parser.add_argument("--loop",     action="store_true",
                        help="Run continuously every 4h")
    parser.add_argument("--register", action="store_true",
                        help="Register on-chain (before Jun 22)")
    parser.add_argument("--serve",    action="store_true",
                        help="Start BNB Agent strategy server")
    parser.add_argument("--status",   action="store_true",
                        help="Show component status")
    parser.add_argument("--dry-run",  type=str, default=None,
                        help="Override DRY_RUN: 'true' or 'false'")
    args = parser.parse_args()

    # Override DRY_RUN — must propagate to EVERY module, not just config.
    # Modules do `from config import DRY_RUN`, which copies the value at import
    # time; mutating only config.DRY_RUN would leave stale copies and produce a
    # dangerous mismatch (banner says PAPER while execution runs LIVE). So we
    # patch the symbol in config AND in every already-imported module that has
    # its own DRY_RUN binding.
    global DRY_RUN
    import config
    if args.dry_run is not None:
        new_val = args.dry_run.lower() == "true"
        config.DRY_RUN = new_val
        DRY_RUN = new_val
        for _modname, _mod in list(sys.modules.items()):
            if _mod is not None and hasattr(_mod, "DRY_RUN") and _mod is not config:
                try:
                    setattr(_mod, "DRY_RUN", new_val)
                except Exception:
                    pass
        print(f"[Agent] DRY_RUN override = {new_val}")

    # Authoritative mode for display = config.DRY_RUN (single source of truth)
    live = not config.DRY_RUN
    print("="*60)
    print("qtrader — BNB Hack AI Trading Agent ⚡")
    print(f"Mode   : {'🔴 LIVE TRADE' if live else '📋 PAPER TRADE'}")
    print(f"Wallet : {AGENT_WALLET_ADDRESS}")
    print(f"Capital: ${INITIAL_CAPITAL:,.0f}")
    print("="*60)

    # ── Load all components ───────────────────────────────────────────────────
    from inference.engine      import QuantumTraderEngine
    from execution.trader      import TWAKTrader
    from execution.twak_client import TWAKClient
    from execution.bnb_agent   import QuantumTraderBNBAgent
    from data.cmc_hub          import CMCAgentHub
    from risk.manager          import RiskManager
    from risk.profit_manager   import ProfitManager
    from alerts                import telegram

    print("\nLoading components...")
    engine    = QuantumTraderEngine()
    trader    = TWAKTrader()
    twak      = TWAKClient()
    cmc_hub   = CMCAgentHub()
    bnb_agent = QuantumTraderBNBAgent()
    risk      = RiskManager()
    profit    = ProfitManager(
        partial_tp_pct  = 0.10,   # +10% → sell 50%
        full_tp_pct     = 0.20,   # +20% → sell 100%
        trailing_stop   = 0.05,   # -5% from peak → sell
        cooldown_cycles = 2,      # 8h cooldown after TP
    )

    telegram.alert_startup(DRY_RUN, AGENT_WALLET_ADDRESS)
    print("All components loaded.\n")

    # ── Special commands ──────────────────────────────────────────────────────
    if args.status:
        show_status(engine, trader, twak, cmc_hub, bnb_agent, risk)
        return

    if args.register:
        print("\n--- On-chain Registration ---")
        # TWAK competition registration
        twak_result = twak.register_competition()
        print(f"TWAK registration: {twak_result}")
        # BNB Agent registration
        bnb_result = bnb_agent.register_on_chain()
        print(f"BNB Agent registration: {bnb_result}")
        return

    if args.serve:
        bnb_agent.start_server()
        return

    # ── Initial data fetch ────────────────────────────────────────────────────
    print("Running initial CMC data fetch...")
    cmc_hub.fetch_all_ohlcv_and_save(hours=1000)
    skills = cmc_hub.run_skills()
    cmc_hub.save_supplementary(skills)
    last_full_fetch = datetime.datetime.now(UTC)
    print(f"Initial fetch complete | x402 paid: ${skills['x402_spent']:.4f}\n")

    # ── Main loop ─────────────────────────────────────────────────────────────
    # Checks every 1h:
    #   - Every 1h: fetch fresh prices, check profit manager (TP/trail stop)
    #   - Every 4h: full inference (phases 1-4 → PPO weights)
    CHECK_INTERVAL_S = 15 * 60        # 15 min price check + profit manager
    INFER_INTERVAL_H = 1             # full PPO inference every 1h

    if args.loop:
        print(f"Price check every 15min | "
              f"Full inference every 1h. Ctrl+C to stop.\n")

        last_inference = datetime.datetime.min.replace(tzinfo=UTC)

        while True:
            now = datetime.datetime.now(UTC)

            # ── Every 15min: fast price check + profit manager ───────────
            # Uses Binance ticker (free, no rate limit) — not CMC API
            try:
                prices = trader.get_current_prices()   # Binance /ticker/price
                profit.update_prices(prices, trader.weights)
                adj_w, adj_cash, tp_actions = profit.apply(
                    trader.weights.copy(), prices, trader.cash_weight)

                if tp_actions:
                    print(f"\n[{now.strftime('%H:%M UTC')}] Profit Manager triggered:")
                    for token, action in tp_actions.items():
                        print(f"  {token}: {action}")
                    # Execute the TP trade immediately
                    execute, reason = risk.should_execute(
                        trader.weights, adj_w,
                        trader.cash_weight, adj_cash, "ok")
                    if execute or tp_actions:
                        trade_result = trader.rebalance(adj_w, adj_cash, "profit_manager")
                        risk.record_trade()
                        telegram.alert_trade(
                            weights      = adj_w,
                            cash_w       = adj_cash,
                            capital      = trade_result["capital"],
                            total_return = (trade_result["capital"] - INITIAL_CAPITAL) / INITIAL_CAPITAL,
                            reason       = f"profit_manager: {list(tp_actions.values())}",
                            tx_hashes    = [],
                        )

                # ── Every 4h: full inference cycle ────────────────────────
                hours_since_infer = (now - last_inference).total_seconds() / 3600
                if hours_since_infer >= INFER_INTERVAL_H:
                    last_inference  = now
                    last_full_fetch = run_cycle(
                        engine, trader, risk, twak,
                        cmc_hub, bnb_agent, profit, last_full_fetch)

            except KeyboardInterrupt:
                print("\nStopped.")
                break
            except Exception as e:
                print(f"[Agent] Error: {e}")
                traceback.print_exc()
                telegram.alert_error(str(e))

            # Sleep 15min then check again
            next_check = now + datetime.timedelta(seconds=CHECK_INTERVAL_S)
            mins_to_infer = max(0, int(
                (INFER_INTERVAL_H * 3600 -
                 (now - last_inference).total_seconds()) / 60))
            print(f"[{now.strftime('%H:%M UTC')}] "
                  f"Next price check: {next_check.strftime('%H:%M')} | "
                  f"Next inference in: {mins_to_infer}min")
            time.sleep(CHECK_INTERVAL_S)
    else:
        # Single run for cron
        try:
            run_cycle(engine, trader, risk, twak,
                      cmc_hub, bnb_agent, profit, last_full_fetch)
        except Exception as e:
            print(f"[Agent] Error: {e}")
            traceback.print_exc()
            telegram.alert_error(str(e))


if __name__ == "__main__":
    main()
