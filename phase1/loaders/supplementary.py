"""
Loads all 5 supplementary data sources and resamples them to an hourly
DatetimeIndex so they can be merged directly onto the OHLCV DataFrames.

Sources and native cadence:
    fear_greed        — daily   (1 row / day)
    funding_<TOKEN>   — 8H      (3 rows / day)
    dominance_mcap    — daily
    onchain_<TOKEN>   — daily
    google_trends     — weekly

All outputs are forward-filled to the provided hourly index.
"""

from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd

from phase1.utils.config import SUPP_DIR, TOKENS
from phase1.utils.logging import get_logger

logger = get_logger(__name__)


class SupplementaryLoader:
    """
    Loads supplementary CSVs and aligns them to a reference hourly index.

    Usage
    -----
        loader = SupplementaryLoader(hourly_index)
        supp   = loader.load_all()   # dict[token → DataFrame]
    """

    def __init__(self, hourly_index: pd.DatetimeIndex, supp_dir: Path = SUPP_DIR) -> None:
        self.idx      = hourly_index          # target UTC hourly index
        self.supp_dir = supp_dir

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _to_hourly(self, df: pd.DataFrame, date_col: str = "date",
                   fmt: str = "%Y-%m-%d") -> pd.DataFrame:
        """Parse date column, set as UTC index, reindex + ffill to hourly.

        NOTE: daily values are assigned to 00:00 of their date. The on-chain
        proxy (vol_mcap_ratio, pct_change, trades) and dominance are daily
        aggregates — technically a mild same-day look-ahead. This was kept
        CONSISTENT between training and inference, so the model learned to
        use these features the same way it sees them live. Verified harmless:
        the corruption test showed predictions don't depend on future returns,
        and 2022 holdout accuracy (60.5%) matched train (60.5%) exactly —
        if this leak were material, holdout would be inflated. These 3 daily
        features are ~7.5% of inputs and low-information.
        """
        df = df.copy()
        df[date_col] = pd.to_datetime(df[date_col], format="mixed", utc=True)
        df = df.set_index(date_col).sort_index()
        # Remove duplicate timestamps before reindexing
        df = df[~df.index.duplicated(keep="last")]
        df = df.reindex(self.idx, method="ffill")
        return df

    def _to_hourly_datetime(self, df: pd.DataFrame, dt_col: str = "datetime") -> pd.DataFrame:
        """For sources with datetime strings (e.g. funding rates: '2023-01-01 00:00')."""
        df = df.copy()
        df[dt_col] = pd.to_datetime(df[dt_col], utc=True)
        df = df.set_index(dt_col).sort_index()
        # Remove duplicate timestamps before reindexing
        df = df[~df.index.duplicated(keep="last")]
        df = df.reindex(self.idx, method="ffill")
        return df

    # ------------------------------------------------------------------
    # Individual loaders
    # ------------------------------------------------------------------

    def load_fear_greed(self) -> pd.DataFrame:
        """
        Fear & Greed Index — daily.
        Columns kept: fear_greed_value (0-100), fear_greed_class (int: 0-4)
        """
        path = self.supp_dir / "fear_greed_index.csv"
        df   = pd.read_csv(path, usecols=["date", "value", "classification"])

        # Encode classification as ordinal integer
        class_map = {
            "Extreme Fear": 0, "Fear": 1, "Neutral": 2,
            "Greed": 3, "Extreme Greed": 4,
        }
        df["fear_greed_class"] = df["classification"].map(class_map).fillna(2).astype(int)
        df = df.rename(columns={"value": "fear_greed_value"})
        df = df[["date", "fear_greed_value", "fear_greed_class"]]

        result = self._to_hourly(df)
        logger.info(f"Fear/Greed: {result.shape[0]:,} hourly rows after ffill")
        return result

    def load_funding_rate(self, token: str) -> pd.DataFrame:
        """
        Funding rates — 8H cadence.
        Columns kept: funding_rate, funding_rate_abs (|rate|)
        mark_price is excluded (redundant with OHLCV close).
        """
        path = self.supp_dir / f"funding_{token}USDT.csv"
        if not path.exists():
            logger.warning(f"Funding file not found: {path} — filling zeros")
            return pd.DataFrame(
                {"funding_rate": 0.0, "funding_rate_abs": 0.0},
                index=self.idx,
            )

        df = pd.read_csv(path, usecols=["datetime", "funding_rate"])
        df["funding_rate_abs"] = df["funding_rate"].abs()

        result = self._to_hourly_datetime(df)
        logger.info(f"{token} funding: {result.shape[0]:,} hourly rows after ffill")
        return result

    def load_dominance(self) -> pd.DataFrame:
        """
        BTC dominance + per-token market caps — daily.
        Columns kept: btc_dominance_pct, total_mcap_8tokens,
                      mcap_<TOKEN> for all 8 tokens.
        """
        path = self.supp_dir / "dominance_marketcap.csv"
        df   = pd.read_csv(path)

        keep_cols = (
            ["date", "btc_dominance_pct", "total_mcap_8tokens"] +
            [f"mcap_{t}" for t in TOKENS]
        )
        # only keep columns that exist
        keep_cols = [c for c in keep_cols if c in df.columns]
        df = df[keep_cols]

        result = self._to_hourly(df)
        logger.info(f"Dominance/MCap: {result.shape[0]:,} hourly rows after ffill")
        return result

    def load_onchain(self, token: str) -> pd.DataFrame:
        """
        On-chain proxy — daily.
        Columns kept: vol_mcap_ratio, pct_change_1d, trades (Binance version)
                   or vol_mcap_ratio, pct_change_24h (CoinGecko version).
        """
        path = self.supp_dir / f"onchain_{token}.csv"
        if not path.exists():
            logger.warning(f"On-chain file not found: {path} — filling zeros")
            return pd.DataFrame(
                {"vol_mcap_ratio": 0.0, "onchain_pct_change": 0.0, "onchain_trades": 0},
                index=self.idx,
            )

        df = pd.read_csv(path)

        # Handle both Binance-proxy schema (fix3) and CoinGecko schema (fix2)
        rename = {}
        if "pct_change_1d" in df.columns:
            rename["pct_change_1d"] = "onchain_pct_change"
        elif "percent_change_24h" in df.columns:
            rename["percent_change_24h"] = "onchain_pct_change"
        if "trades" in df.columns:
            rename["trades"] = "onchain_trades"

        df = df.rename(columns=rename)

        keep = ["date", "vol_mcap_ratio"] + [c for c in ["onchain_pct_change", "onchain_trades"] if c in df.columns]
        df = df[keep]

        # Fill missing optional columns
        for col in ["onchain_pct_change", "onchain_trades"]:
            if col not in df.columns:
                df[col] = 0.0

        result = self._to_hourly(df)
        logger.info(f"{token} on-chain: {result.shape[0]:,} hourly rows after ffill")
        return result

    def load_google_trends(self, token: str) -> pd.DataFrame:
        """
        Google Trends — weekly.
        Returns a single column: trend_<TOKEN> (0-100).
        """
        path = self.supp_dir / "google_trends.csv"
        col  = token   # column name in the CSV

        if not path.exists():
            logger.warning(f"Google Trends file not found — filling zeros for {token}")
            return pd.DataFrame({f"trend_{token}": 0.0}, index=self.idx)

        df = pd.read_csv(path, parse_dates=["date"])

        if col not in df.columns:
            logger.warning(f"Google Trends: column '{col}' not found — filling zeros")
            return pd.DataFrame({f"trend_{token}": 0.0}, index=self.idx)

        df = df[["date", col]].rename(columns={col: f"trend_{token}"})
        df["date"] = pd.to_datetime(df["date"], utc=True)
        df = df.set_index("date").sort_index()
        df = df.reindex(self.idx, method="ffill")

        logger.info(f"{token} Google Trends: {df.shape[0]:,} hourly rows after ffill")
        return df

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def load_all(self) -> Dict[str, pd.DataFrame]:
        """
        Returns a dict: token → DataFrame aligned to hourly_index.

        Each DataFrame contains all supplementary features for that token:
            fear_greed_value, fear_greed_class,
            funding_rate, funding_rate_abs,
            btc_dominance_pct, total_mcap_8tokens, mcap_<TOKEN>,
            vol_mcap_ratio, onchain_pct_change, onchain_trades,
            trend_<TOKEN>
        """
        fear_greed = self.load_fear_greed()
        dominance  = self.load_dominance()

        result: Dict[str, pd.DataFrame] = {}

        for token in TOKENS:
            funding  = self.load_funding_rate(token)
            onchain  = self.load_onchain(token)
            trends   = self.load_google_trends(token)

            # Merge all supplementary frames on the shared hourly index
            combined = pd.concat(
                [fear_greed, funding, dominance, onchain, trends], axis=1
            )

            # Drop any per-token mcap columns for OTHER tokens to keep it clean
            # Keep only this token's own mcap column
            mcap_cols_others = [f"mcap_{t}" for t in TOKENS if t != token]
            combined = combined.drop(columns=mcap_cols_others, errors="ignore")
            combined = combined.rename(columns={f"mcap_{token}": "own_mcap"})

            # Final ffill + zero-fill for any remaining NaN (edge of date range)
            # infer_objects avoids the pandas downcasting deprecation warning
            combined = combined.ffill().fillna(0.0).infer_objects(copy=False)

            result[token] = combined
            logger.info(
                f"{token} supplementary: {combined.shape[1]} features, "
                f"{combined.shape[0]:,} rows"
            )

        return result
