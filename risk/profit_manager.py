"""
Profit Manager
Automatic take-profit and trailing stop logic per token.
Works alongside the PPO agent — overrides weights when profit targets hit.

Rules:
- Partial take-profit at +15%  → sell 50% of position
- Full take-profit at +30%     → sell 100% of position
- Trailing stop at -8% from peak → sell 100%
- Re-entry allowed after cooldown (2 cycles = 8h)
"""

import datetime
import numpy as np
from typing import Dict, Tuple, Optional
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import TRADE_TOKENS

UTC = datetime.timezone.utc


class ProfitManager:

    def __init__(
        self,
        partial_tp_pct:  float = 0.15,   # +15% → sell 50%
        full_tp_pct:     float = 0.30,   # +30% → sell 100%
        trailing_stop:   float = 0.08,   # -8% from peak → sell 100%
        cooldown_cycles: int   = 2,      # cycles before re-entry after TP
    ):
        self.partial_tp   = partial_tp_pct
        self.full_tp      = full_tp_pct
        self.trailing_stop = trailing_stop
        self.cooldown     = cooldown_cycles

        # Per-token state
        self._entry_price:  Dict[str, float] = {}   # price when position opened
        self._peak_price:   Dict[str, float] = {}   # highest price since entry
        self._tp_hits:      Dict[str, int]   = {}   # how many TPs hit
        self._cooldown_rem: Dict[str, int]   = {}   # cycles remaining in cooldown
        self._partial_done: Dict[str, bool]  = {}   # partial TP already taken

    def update_entry(self, token: str, price: float, weight: float):
        """Record entry when we open or increase a position."""
        if weight > 0.01 and token not in self._entry_price:
            self._entry_price[token]  = price
            self._peak_price[token]   = price
            self._partial_done[token] = False
            self._cooldown_rem[token] = 0

    def update_prices(self, prices: Dict[str, float], weights: np.ndarray):
        """Update peak prices for all open positions."""
        for i, token in enumerate(TRADE_TOKENS):
            w = float(weights[i])
            if w > 0.01 and token in self._peak_price:
                price = prices.get(token, 0)
                if price > self._peak_price.get(token, 0):
                    self._peak_price[token] = price

    def apply(
        self,
        weights:  np.ndarray,
        prices:   Dict[str, float],
        cash_w:   float,
    ) -> Tuple[np.ndarray, float, Dict]:
        """
        Apply take-profit and trailing stop rules to proposed weights.

        Returns:
            adjusted_weights, adjusted_cash, actions_taken
        """
        adj_weights = weights.copy()
        actions     = {}

        for i, token in enumerate(TRADE_TOKENS):
            w     = float(adj_weights[i])
            price = prices.get(token, 0)
            if price <= 0:
                continue

            # ── Cooldown check ────────────────────────────────────────────
            if self._cooldown_rem.get(token, 0) > 0:
                self._cooldown_rem[token] -= 1
                # Block re-entry during cooldown
                if w > 0.01:
                    adj_weights[i] = 0.0
                    actions[token] = f"cooldown ({self._cooldown_rem[token]} cycles left)"
                continue

            # ── Update entry price ────────────────────────────────────────
            if w > 0.01:
                self.update_entry(token, price, w)

            entry = self._entry_price.get(token)
            peak  = self._peak_price.get(token)

            if not entry or entry <= 0 or w < 0.01:
                # No open position — clear state
                if token in self._entry_price and w < 0.01:
                    del self._entry_price[token]
                    if token in self._peak_price: del self._peak_price[token]
                    if token in self._partial_done: del self._partial_done[token]
                continue

            pnl_pct      = (price - entry) / entry
            trail_pct    = (price - peak)  / peak if peak else 0

            # ── Full take-profit ──────────────────────────────────────────
            if pnl_pct >= self.full_tp:
                freed = adj_weights[i]
                adj_weights[i] = 0.0
                cash_w += freed
                self._cooldown_rem[token] = self.cooldown
                del self._entry_price[token]
                if token in self._peak_price: del self._peak_price[token]
                actions[token] = f"FULL TP +{pnl_pct:.1%} (sold 100%)"
                print(f"  [Profit] 🎯 FULL TP {token}: +{pnl_pct:.1%} → sold 100%")
                continue

            # ── Partial take-profit ───────────────────────────────────────
            if pnl_pct >= self.partial_tp and not self._partial_done.get(token):
                sell_half = adj_weights[i] * 0.5
                adj_weights[i] -= sell_half
                cash_w += sell_half
                self._partial_done[token] = True
                # Update entry to current price (cost basis reset)
                self._entry_price[token] = price
                actions[token] = f"PARTIAL TP +{pnl_pct:.1%} (sold 50%)"
                print(f"  [Profit] 💰 PARTIAL TP {token}: +{pnl_pct:.1%} → sold 50%")
                continue

            # ── Trailing stop ─────────────────────────────────────────────
            if trail_pct <= -self.trailing_stop:
                freed = adj_weights[i]
                adj_weights[i] = 0.0
                cash_w += freed
                self._cooldown_rem[token] = self.cooldown
                del self._entry_price[token]
                if token in self._peak_price: del self._peak_price[token]
                actions[token] = f"TRAIL STOP {trail_pct:.1%} from peak (sold 100%)"
                print(f"  [Profit] 🛑 TRAIL STOP {token}: {trail_pct:.1%} from peak")
                continue

        # Clip cash to [0, 1]
        cash_w = float(np.clip(cash_w, 0.0, 1.0))

        # Re-normalize asset weights to fit remaining invested fraction
        invested = 1.0 - cash_w
        total_w  = adj_weights.sum()
        if total_w > 0 and invested > 0:
            adj_weights = adj_weights / total_w * invested
        elif total_w > 0:
            adj_weights = np.zeros_like(adj_weights)

        return adj_weights.astype(np.float32), cash_w, actions

    def get_status(self, prices: Dict[str, float]) -> Dict:
        """Return current P&L per open position."""
        status = {}
        for token in TRADE_TOKENS:
            entry = self._entry_price.get(token)
            peak  = self._peak_price.get(token)
            price = prices.get(token, 0)
            if entry and entry > 0 and price > 0:
                pnl_pct   = (price - entry) / entry
                trail_pct = (price - peak) / peak if peak else 0
                status[token] = {
                    "entry":      round(entry, 4),
                    "current":    round(price, 4),
                    "peak":       round(peak, 4) if peak else price,
                    "pnl":        f"{pnl_pct:+.2%}",
                    "trail":      f"{trail_pct:+.2%}",
                    "partial_tp": self._partial_done.get(token, False),
                    "cooldown":   self._cooldown_rem.get(token, 0),
                    "next_full_tp":    f"+{self.full_tp:.0%}",
                    "next_partial_tp": f"+{self.partial_tp:.0%}",
                    "trail_stop":      f"-{self.trailing_stop:.0%} from peak",
                }
        return status
