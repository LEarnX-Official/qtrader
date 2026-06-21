"""
Loads Binance 1h OHLCV CSVs from data/raw/ into clean DataFrames.

Each CSV has columns:
    open_time, open, high, low, close, volume,
    close_time, quote_volume, trades,
    taker_buy_base, taker_buy_quote, ignore

Output DataFrame columns (per token):
    Open, High, Low, Close, Volume, QuoteVolume, Trades,
    TakerBuyBase, TakerBuyQuote
Index: UTC DatetimeIndex (hourly)
"""

from pathlib import Path
from typing import Dict

import pandas as pd

from phase1.utils.config import RAW_DIR, START_DATE, END_DATE, TOKENS
from phase1.utils.logging import get_logger

logger = get_logger(__name__)

_COL_MAP = {
    "open":             "Open",
    "high":             "High",
    "low":              "Low",
    "close":            "Close",
    "volume":           "Volume",
    "quote_volume":     "QuoteVolume",
    "trades":           "Trades",
    "taker_buy_base":   "TakerBuyBase",
    "taker_buy_quote":  "TakerBuyQuote",
}


class CryptoDataLoader:
    """
    Loads and validates Binance 1h OHLCV data for all tokens.

    Steps:
        1. Read CSV, parse open_time → UTC DatetimeIndex
        2. Keep only the 9 useful columns, cast to float/int
        3. Forward-fill the single 2h gap (2023-03-24 Binance maintenance)
        4. Slice to configured date range
        5. Validate: zero closes, length check
    """

    def __init__(
        self,
        raw_dir: Path = RAW_DIR,
        tokens: list = TOKENS,
        start: str = START_DATE,
        end: str = END_DATE,
    ) -> None:
        self.raw_dir = raw_dir
        self.tokens  = tokens
        self.start   = pd.Timestamp(start, tz="UTC")
        self.end     = pd.Timestamp(end,   tz="UTC") + pd.Timedelta(hours=23)

    # ------------------------------------------------------------------

    def load_token(self, token: str) -> pd.DataFrame:
        path = self.raw_dir / f"{token}USDT_1h.csv"
        if not path.exists():
            logger.error(f"{token}: file not found at {path}")
            return pd.DataFrame()

        df = pd.read_csv(path, dtype=str)

        # Parse timestamp (ms) → UTC datetime index
        df.index = pd.to_datetime(df["open_time"].astype(float), unit="ms", utc=True)
        df.index.name = "datetime"

        # Keep useful columns
        keep = list(_COL_MAP.keys())
        df = df[keep].rename(columns=_COL_MAP)
        df = df.astype(float)
        df["Trades"] = df["Trades"].astype(int)

        # Slice to configured range
        df = df.loc[self.start : self.end]

        # Build a clean continuous hourly index and forward-fill gaps
        full_idx = pd.date_range(df.index[0], df.index[-1], freq="1h", tz="UTC")
        df = df.reindex(full_idx)
        n_gaps = df["Close"].isna().sum()
        if n_gaps:
            df.ffill(inplace=True)
            logger.info(f"{token}: forward-filled {n_gaps} missing hourly bars")

        # Sanity checks
        zero_closes = (df["Close"] == 0).sum()
        if zero_closes:
            logger.warning(f"{token}: {zero_closes} zero close prices")

        logger.info(
            f"{token}: {len(df):,} candles | "
            f"{df.index[0].date()} → {df.index[-1].date()}"
        )
        return df

    def load_all(self) -> Dict[str, pd.DataFrame]:
        data = {}
        for token in self.tokens:
            df = self.load_token(token)
            if not df.empty:
                data[token] = df
        logger.info(f"Loaded {len(data)}/{len(self.tokens)} tokens")
        return data

    def quality_report(self, data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        if not data:
            return pd.DataFrame(columns=["token", "candles", "start", "end",
                                         "zero_closes", "nan_any", "price_min",
                                         "price_max", "ok"])
        rows = []
        for token, df in data.items():
            rows.append({
                "token":       token,
                "candles":     len(df),
                "start":       df.index[0].date(),
                "end":         df.index[-1].date(),
                "zero_closes": int((df["Close"] == 0).sum()),
                "nan_any":     bool(df.isna().any().any()),
                "price_min":   df["Close"].min(),
                "price_max":   df["Close"].max(),
            })
        report = pd.DataFrame(rows)
        report["ok"] = (report["zero_closes"] == 0) & (~report["nan_any"])
        return report
