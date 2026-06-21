"""
Pure pandas/numpy technical indicator library — no TA-Lib dependency.
All functions accept pd.Series and return pd.Series with the same index.

Carried over from crypto/phase1/features/indicators.py and extended with
crypto-specific indicators.
"""

from typing import Tuple

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Moving averages
# ---------------------------------------------------------------------------

def sma(series: pd.Series, n: int) -> pd.Series:
    return series.rolling(n, min_periods=1).mean()


def ema(series: pd.Series, n: int) -> pd.Series:
    return series.ewm(span=n, adjust=False).mean()


def wilder_smooth(series: pd.Series, n: int) -> pd.Series:
    """Wilder's smoothing — used by RSI, ATR, ADX."""
    result = series.copy().astype(float)
    result.iloc[:n] = np.nan
    if len(series) > n:
        result.iloc[n] = series.iloc[1:n + 1].mean()
        for i in range(n + 1, len(series)):
            result.iloc[i] = (result.iloc[i - 1] * (n - 1) + series.iloc[i]) / n
    return result


# ---------------------------------------------------------------------------
# Momentum
# ---------------------------------------------------------------------------

def rsi(close: pd.Series, n: int = 14) -> pd.Series:
    delta    = close.diff()
    avg_gain = wilder_smooth(delta.clip(lower=0), n)
    avg_loss = wilder_smooth((-delta).clip(lower=0), n)
    rs       = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def macd(
    close: pd.Series,
    fast: int = 12,
    slow: int = 26,
    signal: int = 9,
) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """Returns (macd_line, signal_line, histogram)."""
    line = ema(close, fast) - ema(close, slow)
    sig  = ema(line, signal)
    return line, sig, line - sig


# ---------------------------------------------------------------------------
# Volatility / Bands
# ---------------------------------------------------------------------------

def bollinger(
    close: pd.Series, n: int = 20, k: float = 2.0
) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """Returns (upper, middle, lower)."""
    mid = sma(close, n)
    std = close.rolling(n, min_periods=1).std()
    return mid + k * std, mid, mid - k * std


def true_range(high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
    prev = close.shift(1)
    return pd.concat(
        [high - low, (high - prev).abs(), (low - prev).abs()], axis=1
    ).max(axis=1)


def atr(high: pd.Series, low: pd.Series, close: pd.Series, n: int = 14) -> pd.Series:
    return wilder_smooth(true_range(high, low, close), n)


# ---------------------------------------------------------------------------
# Trend strength
# ---------------------------------------------------------------------------

def adx(
    high: pd.Series, low: pd.Series, close: pd.Series, n: int = 14
) -> pd.Series:
    up_move  = high.diff()
    dn_move  = -low.diff()
    plus_dm  = pd.Series(
        np.where((up_move > dn_move) & (up_move > 0), up_move, 0.0),
        index=high.index,
    )
    minus_dm = pd.Series(
        np.where((dn_move > up_move) & (dn_move > 0), dn_move, 0.0),
        index=high.index,
    )
    _atr      = atr(high, low, close, n)
    plus_di   = 100 * wilder_smooth(plus_dm, n) / _atr.replace(0, np.nan)
    minus_di  = 100 * wilder_smooth(minus_dm, n) / _atr.replace(0, np.nan)
    dx        = (100 * (plus_di - minus_di).abs() / (plus_di + minus_di)).replace(
        [np.inf, -np.inf], np.nan
    )
    return wilder_smooth(dx.fillna(0), n)


def stochastic(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    k_period: int = 14,
    d_period: int = 3,
) -> Tuple[pd.Series, pd.Series]:
    """Returns (%K, %D)."""
    lowest  = low.rolling(k_period, min_periods=1).min()
    highest = high.rolling(k_period, min_periods=1).max()
    pct_k   = 100 * (close - lowest) / (highest - lowest).replace(0, np.nan)
    pct_d   = pct_k.rolling(d_period, min_periods=1).mean()
    return pct_k, pct_d


# ---------------------------------------------------------------------------
# Volatility estimators  (annualised for hourly: sqrt(8760))
# ---------------------------------------------------------------------------

_ANN_H = np.sqrt(8760)   # annualisation factor for hourly returns


def hist_vol(returns: pd.Series, n: int) -> pd.Series:
    """Rolling historical volatility (annualised)."""
    return returns.rolling(n, min_periods=max(2, n // 4)).std() * _ANN_H


def parkinson_vol(high: pd.Series, low: pd.Series, n: int) -> pd.Series:
    hl = np.log(high / low.replace(0, np.nan))
    return (
        np.sqrt(1 / (4 * np.log(2)) * (hl ** 2).rolling(n, min_periods=max(2, n // 4)).mean())
        * _ANN_H
    )


def garman_klass_vol(
    high: pd.Series, low: pd.Series,
    open_: pd.Series, close: pd.Series, n: int
) -> pd.Series:
    hl_sq = (np.log(high / low.replace(0, np.nan))) ** 2
    co_sq = (np.log(close / open_.replace(0, np.nan))) ** 2
    return (
        np.sqrt(
            0.5 * hl_sq.rolling(n, min_periods=max(2, n // 4)).mean()
            - (2 * np.log(2) - 1) * co_sq.rolling(n, min_periods=max(2, n // 4)).mean()
        )
        * _ANN_H
    )


def realized_vol(returns: pd.Series, n: int) -> pd.Series:
    return np.sqrt((returns ** 2).rolling(n, min_periods=max(2, n // 4)).sum()) * _ANN_H


# ---------------------------------------------------------------------------
# Crypto-specific
# ---------------------------------------------------------------------------

def taker_buy_ratio(taker_buy_base: pd.Series, volume: pd.Series) -> pd.Series:
    """
    Fraction of volume initiated by buyers (takers hitting the ask).
    Range: [0, 1].  > 0.5 → buying pressure;  < 0.5 → selling pressure.
    This is only available on Binance kline data.
    """
    return taker_buy_base / volume.replace(0, np.nan)


def rolling_corr_with_btc(
    token_returns: pd.Series,
    btc_returns: pd.Series,
    n: int = 24,
) -> pd.Series:
    """
    Rolling n-hour Pearson correlation of token returns with BTC returns.
    Measures how strongly the token co-moves with BTC (market regime signal).
    """
    return token_returns.rolling(n, min_periods=max(2, n // 2)).corr(btc_returns)


def funding_rate_momentum(funding_rate: pd.Series, n: int = 3) -> pd.Series:
    """
    Rolling mean of the last n funding rate readings (each 8H).
    Captures persistent long/short bias in the perpetual market.
    """
    return funding_rate.rolling(n, min_periods=1).mean()


def volume_profile_ratio(
    quote_volume: pd.Series, n_short: int = 24, n_long: int = 168
) -> pd.Series:
    """
    Short-window USDT volume relative to long-window average.
    Spikes indicate unusual activity (liquidations, news events).
    """
    return (
        quote_volume.rolling(n_short, min_periods=1).mean()
        / quote_volume.rolling(n_long, min_periods=1).mean().replace(0, np.nan)
    )
