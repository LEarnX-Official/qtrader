"""
Phase 5 Trading Environment.

State  : 8 tokens × 565 + 7 weights + 3 metrics = 4530 dims
         (agent OBSERVES BTC's Ψ₀/ΣAᵢ/α/probs as market context)
Action : 7 trade token weights (BTC excluded from portfolio)
Returns: 7 trade tokens (BTC excluded from reward/Kelly)

Including BTC in state gives the agent direct visibility of the dominant
market driver without requiring it to hold BTC.
"""

from typing import Dict, Optional, Tuple
import numpy as np

from phase5.utils.config import (
    BIN_CENTERS, INITIAL_CAPITAL, KELLY_FRACTION, LAMBDA_CALIB,
    LAMBDA_DRAWDOWN, LAMBDA_RETURN, LAMBDA_RISK, LAMBDA_TURNOVER,
    MAX_DRAWDOWN_THR, MAX_POSITION, N_ASSETS, N_STATE_TOKENS,
    PER_ASSET_DIM, STATE_DIM, TRANSACTION_COST, SLIPPAGE,
)
from phase5.utils.logging import get_logger

logger = get_logger(__name__)
BC = np.array(BIN_CENTERS, dtype=np.float32)


class CryptoPortfolioEnv:
    """
    Trading environment.

    Data shapes:
        psi0     : (T, 8, 32)   — all tokens for state
        sigma_ai : (T, 8, 512)
        alpha    : (T, 8, 1)
        probs    : (T, 8, 20)   — BTC probs give regime context
        returns  : (T, 7)       — only trade tokens for reward
    """

    def __init__(
        self,
        psi0:     np.ndarray,   # (T, 8, 32)
        sigma_ai: np.ndarray,   # (T, 8, 512)
        alpha:    np.ndarray,   # (T, 8, 1)
        probs:    np.ndarray,   # (T, 8, 20)
        returns:  np.ndarray,   # (T, 7)
        episode_len: Optional[int] = None,
        initial_capital: float = INITIAL_CAPITAL,
    ):
        self.psi0     = psi0.astype(np.float32)
        self.sigma_ai = sigma_ai.astype(np.float32)
        self.alpha    = alpha.astype(np.float32)
        self.probs    = probs.astype(np.float32)
        self.returns  = returns.astype(np.float32)

        self.T           = returns.shape[0]
        self.N           = N_ASSETS          # 7 — trade tokens
        self.N_state     = N_STATE_TOKENS    # 8 — state observation
        self.episode_len = episode_len
        self.initial_capital = initial_capital
        self.state_dim   = STATE_DIM         # 4530
        self.action_dim  = N_ASSETS          # 7
        self._reset_state()

    def _reset_state(self):
        self.weights      = np.zeros(self.N, dtype=np.float32)
        self.capital      = self.initial_capital
        self.peak_capital = self.initial_capital
        self.port_returns: list = []
        self.port_values:  list = [self.initial_capital]

    def reset(self) -> np.ndarray:
        self._reset_state()
        if self.episode_len and self.episode_len < self.T:
            self._start = np.random.randint(0, self.T - self.episode_len)
        else:
            self._start = 0
        self._end = min(
            self._start + self.episode_len if self.episode_len else self.T,
            self.T
        )
        self.t = self._start
        return self._obs()

    def _obs(self) -> np.ndarray:
        t        = self.t
        pnl      = (self.capital - self.initial_capital) / self.initial_capital
        vol      = float(np.std(self.port_returns[-30:])) if len(self.port_returns) >= 30 else 0.0
        drawdown = (self.peak_capital - self.capital) / max(self.peak_capital, 1e-8)

        # State: 8 tokens × 565 features (BTC included as context)
        return np.concatenate([
            self.psi0[t].flatten(),      # (8×32 = 256)
            self.sigma_ai[t].flatten(),  # (8×512 = 4096)
            self.alpha[t].flatten(),     # (8×1  = 8)
            self.probs[t].flatten(),     # (8×20 = 160)
            self.weights,                # (7) — trade token weights only
            [pnl, vol, drawdown],        # (3)
        ]).astype(np.float32)            # total: 256+4096+8+160+7+3 = 4530 ✓

    def kelly_fractions(self) -> np.ndarray:
        """Kelly fractions for 7 TRADE tokens only (not BTC)."""
        # probs shape: (8, 20) — take only trade token rows (indices 1-7)
        # ALL_TOKENS = [BTC, BNB, SOL, ETH, XRP, INJ, DOGE, LTC]
        # TRADE_TOKENS = [BNB, SOL, ETH, XRP, INJ, DOGE, LTC] → indices 1-7
        p = self.probs[self.t, 1:, :]          # (7, 20) — skip BTC (index 0)
        E = (p * BC).sum(axis=1)               # (7,)
        V = (p * (BC - E[:, None])**2).sum(axis=1)
        kelly = np.where(V > 1e-7, E / V, 0.0)
        return np.clip(kelly * KELLY_FRACTION, 0.0, 1.0).astype(np.float32)

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, Dict]:
        # Action may be (N+1,) with a cash logit appended; extract asset weights.
        # The Dirichlet outputs sum to 1 over (N_assets + 1) dims; the cash
        # weight is action[-1] and the remaining fraction is invested.
        if action.shape[0] == self.N + 1:
            cash_w = float(np.clip(action[-1], 0.0, 1.0))
            action = action[:self.N]
        else:
            cash_w = 0.0

        # Rescale asset weights to invested fraction (1 - cash_w)
        action = np.clip(action, 0.0, MAX_POSITION)
        s = action.sum()
        invested = 1.0 - cash_w
        action = (action / s * invested) if s > 1e-6 else np.full(self.N, invested / self.N)

        turnover  = np.abs(action - self.weights).sum()
        tc        = turnover * TRANSACTION_COST * self.capital
        slip      = turnover * SLIPPAGE * self.capital
        old_w     = self.weights.copy()
        self.weights = action.astype(np.float32)

        # Returns from 7 trade tokens
        asset_ret = self.returns[self.t]           # (7,)
        port_ret  = float((self.weights * asset_ret).sum())
        self.port_returns.append(port_ret)
        self.capital = self.capital * (1.0 + port_ret) - tc - slip
        self.port_values.append(self.capital)
        if self.capital > self.peak_capital:
            self.peak_capital = self.capital

        reward = self._reward(port_ret, turnover, old_w, asset_ret)
        self.t += 1

        # Early termination on ruin — capital dropped >40% from peak
        current_dd = (self.peak_capital - self.capital) / max(self.peak_capital, 1e-8)
        ruin = current_dd > 0.40 or self.capital < self.initial_capital * 0.10
        done = (self.t >= self._end) or ruin
        if ruin:
            reward -= 5.0   # large terminal penalty for ruin

        obs  = self._obs() if not done else np.zeros(self.state_dim, dtype=np.float32)
        info = {
            "capital":      self.capital,
            "port_return":  port_ret,
            "turnover":     turnover,
            "sharpe":       self._sharpe() if done else 0.0,
            "max_drawdown": self._max_drawdown(),
            "total_return": (self.capital / self.initial_capital) - 1.0,
        }
        return obs, reward, done, info

    def _reward(self, port_ret, turnover, old_w, asset_ret) -> float:
        r = port_ret
        if len(self.port_returns) >= 30:
            vol = float(np.std(self.port_returns[-30:]))
            r  -= LAMBDA_RISK * vol**2
        r -= LAMBDA_TURNOVER * turnover
        dd = (self.peak_capital - self.capital) / max(self.peak_capital, 1e-8)
        if dd > MAX_DRAWDOWN_THR:
            r -= LAMBDA_DRAWDOWN * (dd - MAX_DRAWDOWN_THR)

        # Cumulative return bonus: nudges agent toward positive absolute returns
        pnl = (self.capital - self.initial_capital) / self.initial_capital
        r += LAMBDA_RETURN * pnl

        # Calibration bonus using 7 trade token probs (skip BTC at index 0)
        p_dist = self.probs[self.t - 1, 1:, :]    # (7, 20)
        calib  = 0.0
        for i in range(self.N):
            if old_w[i] > 0.01:
                idx    = int(np.clip(np.searchsorted(BC, asset_ret[i]) - 1, 0, 19))
                calib += float(np.log(p_dist[i, idx] + 1e-8))
        r += LAMBDA_CALIB * calib
        return float(r)

    def _sharpe(self) -> float:
        r = np.array(self.port_returns)
        if len(r) < 2: return 0.0
        std = r.std()
        return float(r.mean() / std * np.sqrt(365 * 6)) if std > 1e-9 else 0.0

    def _max_drawdown(self) -> float:
        v     = np.array(self.port_values)
        peaks = np.maximum.accumulate(v)
        return float(((peaks - v) / np.maximum(peaks, 1e-8)).max())

    def equal_weight_sharpe(self) -> float:
        r   = self.returns[self._start:self._end].mean(axis=1)
        std = r.std()
        return float(r.mean() / std * np.sqrt(365 * 6)) if std > 1e-9 else 0.0
