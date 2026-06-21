"""
TWAK Execution Layer
Handles on-chain BSC swaps via Trust Wallet Agent Kit.
In DRY_RUN mode: logs trades only, no real transactions.
In LIVE mode: executes real BEP-20 swaps via TWAK signing.
"""

import json
import datetime
import requests
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import (
    TWAK_API_KEY, AGENT_WALLET_ADDRESS, WALLET_PASSWORD,
    TOKEN_ADDRESSES, BASE_CURRENCY, BASE_ADDRESS, TRADE_TOKENS,
    BSC_RPC_URL, BSC_CHAIN_ID, PANCAKESWAP_ROUTER,
    COMPETITION_CONTRACT, DRY_RUN, RESULTS_DIR,
    INITIAL_CAPITAL, TRANSACTION_COST, SLIPPAGE,
)

UTC = datetime.timezone.utc


class TWAKTrader:
    """
    Executes portfolio rebalancing trades on BSC via TWAK.
    DRY_RUN=True  → paper trade, log only
    DRY_RUN=False → live trade on BSC
    """

    def __init__(self):
        self.dry_run        = DRY_RUN
        self.capital        = INITIAL_CAPITAL
        self.holdings       = {BASE_CURRENCY: INITIAL_CAPITAL}  # token → USD value
        self.weights        = np.zeros(7, dtype=np.float32)
        self.cash_weight    = 1.0
        self.trade_history  = []
        self.port_returns   = []
        self.peak_capital   = INITIAL_CAPITAL

        mode = "PAPER TRADE" if self.dry_run else "LIVE TRADE"
        print(f"[Trader] Mode: {mode} | Wallet: {AGENT_WALLET_ADDRESS[:12]}...")

        if not self.dry_run:
            self._init_twak()

    def _init_twak(self):
        """Initialize TWAK connection for live trading."""
        try:
            import subprocess
            result = subprocess.run(
                ["twak", "status"], capture_output=True, text=True, timeout=10)
            print(f"[TWAK] Status: {result.stdout.strip()}")
        except Exception as e:
            print(f"[TWAK] Init warning: {e}")

    # ── Portfolio Valuation ───────────────────────────────────────────────────

    def get_current_prices(self) -> Dict[str, float]:
        """Fetch current prices from Binance for all trade tokens."""
        prices = {}
        for token in TRADE_TOKENS:
            try:
                r = requests.get(
                    f"https://api.binance.com/api/v3/ticker/price",
                    params={"symbol": f"{token}USDT"}, timeout=10)
                prices[token] = float(r.json()["price"])
            except Exception as e:
                print(f"  [Trader] Price fetch {token}: {e}")
                prices[token] = 0.0
        return prices

    def portfolio_value(self, prices: Dict[str, float]) -> float:
        """Calculate current total portfolio value in USD."""
        total = self.holdings.get(BASE_CURRENCY, 0.0)
        for token, price in prices.items():
            if token in self.holdings and price > 0:
                total += self.holdings[token] * price
        return total

    # ── Trade Execution ───────────────────────────────────────────────────────

    def _calc_target_holdings(
        self,
        weights:   np.ndarray,
        cash_w:    float,
        capital:   float,
        prices:    Dict[str, float],
    ) -> Dict[str, float]:
        """
        Convert portfolio weights to target token holdings.
        Returns dict: token → amount in tokens (not USD)
        """
        targets = {BASE_CURRENCY: capital * cash_w}
        for i, token in enumerate(TRADE_TOKENS):
            usd_value = capital * float(weights[i])
            if prices.get(token, 0) > 0:
                targets[token] = usd_value / prices[token]
            else:
                targets[token] = 0.0
        return targets

    def _calc_trades(
        self,
        current: Dict[str, float],
        target:  Dict[str, float],
        prices:  Dict[str, float],
    ) -> List[Dict]:
        """
        Calculate required buy/sell trades to move from current to target.
        Returns list of trade dicts: {token, action, amount_token, amount_usd}
        """
        trades = []
        for token in TRADE_TOKENS:
            curr_amount = current.get(token, 0.0)
            tgt_amount  = target.get(token, 0.0)
            diff        = tgt_amount - curr_amount
            price       = prices.get(token, 0.0)
            if price <= 0:
                continue
            diff_usd = abs(diff) * price
            if diff_usd < 1.0:   # skip sub-$1 trades
                continue
            trades.append({
                "token":        token,
                "action":       "buy" if diff > 0 else "sell",
                "amount_token": abs(diff),
                "amount_usd":   diff_usd,
                "price":        price,
            })
        return trades

    def _execute_twak_swap(self, trade: Dict) -> Optional[str]:
        """
        Execute a single swap via TWAK on BSC.
        Returns tx_hash if successful, None if failed.
        """
        try:
            import subprocess, json

            token_addr = TOKEN_ADDRESSES.get(trade["token"])
            if not token_addr:
                print(f"  [TWAK] No address for {trade['token']}")
                return None

            if trade["action"] == "buy":
                # BASE_CURRENCY → Token
                cmd = [
                    "twak", "swap",
                    "--from", BASE_ADDRESS,
                    "--to",   token_addr,
                    "--amount", str(round(trade["amount_usd"], 2)),
                    "--slippage", "0.5",
                    "--wallet", AGENT_WALLET_ADDRESS,
                ]
            else:
                # Token → BASE_CURRENCY
                cmd = [
                    "twak", "swap",
                    "--from", token_addr,
                    "--to",   BASE_ADDRESS,
                    "--amount", str(round(trade["amount_token"], 6)),
                    "--slippage", "0.5",
                    "--wallet", AGENT_WALLET_ADDRESS,
                ]

            result = subprocess.run(
                cmd, capture_output=True, text=True, timeout=60)

            if result.returncode == 0:
                output = json.loads(result.stdout)
                tx_hash = output.get("txHash", "")
                print(f"  [TWAK] {trade['action'].upper()} {trade['token']} "
                      f"${trade['amount_usd']:.2f} → tx: {tx_hash[:16]}...")
                return tx_hash
            else:
                print(f"  [TWAK] Swap failed: {result.stderr}")
                return None

        except Exception as e:
            print(f"  [TWAK] Execute error: {e}")
            return None

    def rebalance(
        self,
        weights:  np.ndarray,
        cash_w:   float,
        reason:   str = "rebalance",
    ) -> Dict:
        """
        Execute portfolio rebalancing.
        DRY_RUN=True → simulate only
        DRY_RUN=False → execute on BSC via TWAK
        """
        now    = datetime.datetime.now(UTC)
        prices = self.get_current_prices()

        # Calculate current and target portfolio
        old_capital  = self.portfolio_value(prices)
        self.capital = old_capital

        target_holdings = self._calc_target_holdings(
            weights, cash_w, old_capital, prices)
        trades = self._calc_trades(self.holdings, target_holdings, prices)

        if not trades:
            print(f"  [Trader] No significant trades needed")
            return {"trades": [], "capital": old_capital, "reason": reason}

        tx_hashes = []

        if self.dry_run:
            # Paper trade — simulate execution with costs
            for t in trades:
                cost = t["amount_usd"] * (TRANSACTION_COST + SLIPPAGE)
                self.capital -= cost
                print(f"  [Trader] PAPER {t['action'].upper()} "
                      f"{t['token']} ${t['amount_usd']:.2f} "
                      f"(cost: ${cost:.2f})")
            # Update simulated holdings
            self.holdings = {
                BASE_CURRENCY: old_capital * cash_w,
            }
            for i, token in enumerate(TRADE_TOKENS):
                usd_val = old_capital * float(weights[i])
                if prices.get(token, 0) > 0:
                    self.holdings[token] = usd_val / prices[token]
        else:
            # Live trade — execute via TWAK
            for t in trades:
                tx = self._execute_twak_swap(t)
                if tx:
                    tx_hashes.append({"token": t["token"], "tx": tx})
                    # Update holdings after confirmed tx
                    if t["action"] == "buy":
                        self.holdings[t["token"]] = (
                            self.holdings.get(t["token"], 0) + t["amount_token"])
                        self.holdings[BASE_CURRENCY] = (
                            self.holdings.get(BASE_CURRENCY, 0) - t["amount_usd"])
                    else:
                        self.holdings[t["token"]] = max(
                            0, self.holdings.get(t["token"], 0) - t["amount_token"])
                        self.holdings[BASE_CURRENCY] = (
                            self.holdings.get(BASE_CURRENCY, 0) + t["amount_usd"])

        # Update state
        self.weights     = weights.copy()
        self.cash_weight = cash_w
        new_capital      = self.portfolio_value(prices)
        port_ret         = (new_capital - old_capital) / max(old_capital, 1e-8)
        self.port_returns.append(port_ret)
        if new_capital > self.peak_capital:
            self.peak_capital = new_capital

        # Log trade
        trade_record = {
            "datetime":     now.isoformat(),
            "capital":      round(new_capital, 2),
            "port_return":  round(port_ret, 6),
            "cash_weight":  round(cash_w, 4),
            "reason":       reason,
            "dry_run":      self.dry_run,
            "tx_hashes":    tx_hashes,
            "n_trades":     len(trades),
            **{f"w_{t}": round(float(weights[i]), 4)
               for i, t in enumerate(TRADE_TOKENS)},
            **{f"price_{t}": prices.get(t, 0)
               for t in TRADE_TOKENS},
        }
        self.trade_history.append(trade_record)
        self._save_trade(trade_record)

        total_return = (new_capital - INITIAL_CAPITAL) / INITIAL_CAPITAL
        max_dd = self._max_drawdown()

        print(f"  [Trader] Capital: ${new_capital:,.2f} | "
              f"Return: {total_return:+.2%} | "
              f"MaxDD: {max_dd:.2%} | "
              f"Trades: {len(trades)}")

        return {
            "trades":        trades,
            "tx_hashes":     tx_hashes,
            "capital":       new_capital,
            "total_return":  total_return,
            "max_drawdown":  max_dd,
            "reason":        reason,
        }

    def _save_trade(self, record: Dict):
        """Append trade record to results CSV."""
        path = RESULTS_DIR / "trades_live.csv"
        df   = pd.DataFrame([record])
        if path.exists():
            df.to_csv(path, mode="a", header=False, index=False)
        else:
            df.to_csv(path, index=False)

    def _max_drawdown(self) -> float:
        if not self.trade_history:
            return 0.0
        capitals = [t["capital"] for t in self.trade_history]
        capitals = [INITIAL_CAPITAL] + capitals
        v     = np.array(capitals)
        peaks = np.maximum.accumulate(v)
        return float(((peaks - v) / np.maximum(peaks, 1e-8)).max())

    # ── Competition Registration ───────────────────────────────────────────────

    def register_for_competition(self) -> bool:
        """
        Register agent wallet on-chain for Track 1 competition.
        Must be called before June 22, 2026.
        """
        print(f"\n[TWAK] Registering for competition...")
        print(f"  Contract: {COMPETITION_CONTRACT}")
        print(f"  Wallet:   {AGENT_WALLET_ADDRESS}")

        if self.dry_run:
            print("  [DRY RUN] Would execute: twak compete register")
            print("  [DRY RUN] Set DRY_RUN=False to actually register on-chain")
            return False

        try:
            import subprocess
            result = subprocess.run(
                ["twak", "compete", "register",
                 "--contract", COMPETITION_CONTRACT,
                 "--wallet", AGENT_WALLET_ADDRESS],
                capture_output=True, text=True, timeout=60)

            if result.returncode == 0:
                print(f"  ✓ Registered! Output: {result.stdout}")
                return True
            else:
                print(f"  ✗ Registration failed: {result.stderr}")
                return False
        except Exception as e:
            print(f"  ✗ Registration error: {e}")
            return False

    def get_portfolio_summary(self) -> Dict:
        prices = self.get_current_prices()
        capital = self.portfolio_value(prices)
        total_return = (capital - INITIAL_CAPITAL) / INITIAL_CAPITAL
        return {
            "capital":       round(capital, 2),
            "total_return":  f"{total_return:+.2%}",
            "max_drawdown":  f"{self._max_drawdown():.2%}",
            "n_trades":      len(self.trade_history),
            "holdings":      {k: round(v, 6) for k, v in self.holdings.items()},
            "weights":       {t: round(float(self.weights[i]), 4)
                              for i, t in enumerate(TRADE_TOKENS)},
            "cash":          round(self.cash_weight, 4),
            "dry_run":       self.dry_run,
        }
