"""
Risk Manager
Enforces all risk rules before any trade is executed.
- 10% drawdown → warning + reduce position sizes 50%
- 15% drawdown → hard stop, go 100% cash
- 30% drawdown → competition disqualification threshold (never reach)
- Min 1 trade per day → forced micro-rebalance if needed
- Max position 40% per asset
"""

import datetime
import numpy as np
from pathlib import Path
from typing import Dict, Tuple, Optional
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import (
    DD_WARN_THRESHOLD, DD_STOP_THRESHOLD, DD_COMPETITION_CAP,
    MIN_TRADES_PER_DAY, FORCE_TRADE_HOURS, MAX_POSITION,
    MIN_TRADE_THRESHOLD, TRADE_TOKENS,
)

UTC = datetime.timezone.utc


class RiskManager:

    def __init__(self):
        self.peak_capital    = None   # set on first call
        self.trade_log       = []     # list of datetime of each trade
        self.stopped         = False  # hard stop flag
        self.reduced         = False  # 50% reduction flag
        self.start_time      = datetime.datetime.now(UTC)  # when agent started

    def update_peak(self, capital: float):
        if self.peak_capital is None or capital > self.peak_capital:
            self.peak_capital = capital

    def current_drawdown(self, capital: float) -> float:
        if self.peak_capital is None or self.peak_capital <= 0:
            return 0.0
        return (self.peak_capital - capital) / self.peak_capital

    def record_trade(self):
        self.trade_log.append(datetime.datetime.now(UTC))

    def trades_today(self) -> int:
        today = datetime.datetime.now(UTC).date()
        return sum(1 for t in self.trade_log if t.date() == today)

    def hours_since_last_trade(self) -> float:
        # If no trade yet, measure from agent start time (not infinity).
        # This prevents a forced trade firing on the very first cycle.
        if not self.trade_log:
            delta = datetime.datetime.now(UTC) - self.start_time
            return delta.total_seconds() / 3600
        last  = max(self.trade_log)
        delta = datetime.datetime.now(UTC) - last
        return delta.total_seconds() / 3600

    def check(
        self,
        capital:  float,
        weights:  np.ndarray,   # (7,) proposed weights
        cash_w:   float,
    ) -> Tuple[np.ndarray, float, str]:
        """
        Apply all risk rules to proposed weights.

        Returns:
            (adjusted_weights, adjusted_cash, status_message)
            status: "ok" | "warn" | "stop" | "forced_trade"
        """
        self.update_peak(capital)
        dd = self.current_drawdown(capital)

        # ── Hard stop ────────────────────────────────────────────────────────
        if dd >= DD_STOP_THRESHOLD:
            self.stopped = True
            msg = (f"HARD STOP: drawdown={dd:.1%} >= {DD_STOP_THRESHOLD:.0%} threshold. "
                   f"Going 100% cash.")
            return np.zeros(7, dtype=np.float32), 1.0, "stop"

        # ── Already stopped ───────────────────────────────────────────────────
        if self.stopped:
            # Only restart if drawdown recovered below warn threshold
            if dd < DD_WARN_THRESHOLD * 0.5:
                self.stopped = False
                print("  [Risk] Drawdown recovered — resuming trading")
            else:
                return np.zeros(7, dtype=np.float32), 1.0, "stop"

        # ── Warning: reduce sizes 50% ─────────────────────────────────────────
        if dd >= DD_WARN_THRESHOLD:
            self.reduced = True
            weights = weights * 0.5
            cash_w  = 1.0 - weights.sum()
            status  = "warn"
        else:
            self.reduced = False
            status = "ok"

        # ── Max position cap ──────────────────────────────────────────────────
        weights = np.clip(weights, 0.0, MAX_POSITION)
        total   = weights.sum()
        if total > (1.0 - cash_w):
            weights = weights / total * (1.0 - cash_w)

        # ── Minimum trade check ───────────────────────────────────────────────
        hours_since = self.hours_since_last_trade()
        if hours_since >= FORCE_TRADE_HOURS and self.trades_today() == 0:
            # Force a micro-rebalance to meet the 1 trade/day requirement.
            # Only use ELIGIBLE token indices (BNB/SOL are not eligible).
            from config import TRADE_TOKENS, ELIGIBLE_TRADE_TOKENS
            elig_idx = [TRADE_TOKENS.index(t) for t in ELIGIBLE_TRADE_TOKENS
                        if t in TRADE_TOKENS]

            # Among eligible tokens, find which already hold weight
            elig_held = [i for i in elig_idx if weights[i] > 0.005]

            if len(elig_held) >= 2:
                # Swap 0.5% between two eligible holdings
                w_sorted = sorted(elig_held, key=lambda i: weights[i])
                weights[w_sorted[-1]] += 0.005
                weights[w_sorted[0]]  -= 0.005
                weights = np.clip(weights, 0.0, MAX_POSITION)
            elif len(elig_held) == 1:
                # Trim 0.5% from the single holding → cash (counts as a trade)
                weights[elig_held[0]] = max(0.0, weights[elig_held[0]] - 0.005)
                cash_w = min(1.0, cash_w + 0.005)
            else:
                # All cash — buy 1% of the FIRST eligible token (e.g. ETH)
                target = elig_idx[0] if elig_idx else 0
                weights[target] = 0.01
                cash_w = max(0.0, cash_w - 0.01)
            status = "forced_trade"
            print(f"  [Risk] FORCED TRADE — {hours_since:.0f}h since last "
                  f"trade (eligible token only)")

        return weights.astype(np.float32), float(cash_w), status

    def should_execute(
        self,
        old_weights: np.ndarray,
        new_weights: np.ndarray,
        old_cash:    float,
        new_cash:    float,
        status:      str,
    ) -> Tuple[bool, str]:
        """
        Decide whether to actually execute the trade.
        Returns (execute: bool, reason: str)
        """
        if status == "stop":
            return True, "hard_stop_liquidate"

        if status == "forced_trade":
            return True, "min_daily_trade"

        # Check if change is significant enough to be worth the gas
        weight_change = np.abs(new_weights - old_weights).sum()
        cash_change   = abs(new_cash - old_cash)
        total_change  = weight_change + cash_change

        if total_change < MIN_TRADE_THRESHOLD:
            return False, f"change too small ({total_change:.2%} < {MIN_TRADE_THRESHOLD:.0%})"

        return True, "rebalance"

    def get_status_report(self, capital: float) -> Dict:
        dd = self.current_drawdown(capital)
        return {
            "capital":           round(capital, 2),
            "peak_capital":      round(self.peak_capital or capital, 2),
            "drawdown":          round(dd, 4),
            "drawdown_pct":      f"{dd:.2%}",
            "stopped":           self.stopped,
            "reduced":           self.reduced,
            "trades_today":      self.trades_today(),
            "hours_since_trade": round(self.hours_since_last_trade(), 1),
            "warn_at":           f"{DD_WARN_THRESHOLD:.0%}",
            "stop_at":           f"{DD_STOP_THRESHOLD:.0%}",
            "dq_at":             f"{DD_COMPETITION_CAP:.0%}",
        }
