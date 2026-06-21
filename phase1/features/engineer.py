"""
Phase 1 Feature Engineering for Quantum Trader.

Builds a 41-feature matrix per token at 1h resolution, then creates
sliding-window sequences for VAE training.

Feature groups (41 total):
    price       (7)  — log return, HL range, CO ratio, intraday patterns,
                        MA deviations (168h / 720h)
    volume      (5)  — volume change, MA ratio, OBV ratio, quote vol ratio,
                        taker buy ratio
    technical   (15) — RSI, MACD×3, Bollinger×3, ATR, ADX, Stochastic×2,
                        SMA168/720 deviations
    volatility  (4)  — hist vol short/long, Parkinson, Garman-Klass
    crypto      (5)  — funding rate, funding momentum, fear/greed×2,
                        BTC dominance, vol/mcap ratio
    cross_token (1)  — rolling 24h correlation with BTC
    sentiment   (4)  — fear/greed value, fear/greed class,
                        Google Trend, on-chain pct change
    ─────────────────────────────────────────────
    Total             41
"""

from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from phase1.features import indicators as ta
from phase1.utils.config import (
    ADX_PERIOD, ATR_PERIOD, BB_PERIOD, CORR_WINDOW,
    MACD_FAST, MACD_SLOW, MACD_SIG,
    MA_LONG, MA_SHORT, NORM_WINDOW,
    RSI_PERIOD, SEQUENCE_LENGTH, FORECAST_HORIZON,
    STOCH_D, STOCH_K, TOKENS,
    VOL_LONG, VOL_SHORT,
)
from phase1.utils.logging import get_logger

logger = get_logger(__name__)

# Expected feature count — used for assertion
N_FEATURES = 40


class FeatureEngineer:
    """
    Constructs X(token, t) feature matrices and sliding-window sequences.

    Parameters
    ----------
    ohlcv_data   : dict[token → OHLCV DataFrame with hourly UTC index]
    supp_data    : dict[token → supplementary DataFrame (same hourly index)]
    btc_returns  : pre-computed BTC log-returns Series (for cross-token corr)
    """

    def __init__(
        self,
        ohlcv_data: Dict[str, pd.DataFrame],
        supp_data:  Dict[str, pd.DataFrame],
    ) -> None:
        self.ohlcv   = ohlcv_data
        self.supp    = supp_data
        self.features: Dict[str, pd.DataFrame] = {}

        # Pre-compute BTC log-returns once (used for cross-token correlation)
        if "BTC" in ohlcv_data:
            self._btc_ret = np.log(
                ohlcv_data["BTC"]["Close"] / ohlcv_data["BTC"]["Close"].shift(1)
            )
        else:
            self._btc_ret = None

    # ------------------------------------------------------------------
    # Feature groups
    # ------------------------------------------------------------------

    def _price_features(self, df: pd.DataFrame) -> pd.DataFrame:
        feat = pd.DataFrame(index=df.index)
        c, h, lo, o = df["Close"], df["High"], df["Low"], df["Open"]

        feat["log_return"]      = np.log(c / c.shift(1))
        feat["log_hl_range"]    = np.log(h / lo.replace(0, np.nan))
        feat["log_co"]          = np.log(c / o.replace(0, np.nan))
        feat["high_open_ratio"] = (h - o) / o.replace(0, np.nan)
        feat["open_low_ratio"]  = (o - lo) / o.replace(0, np.nan)
        feat["ma_short_dev"]    = c / ta.sma(c, MA_SHORT).replace(0, np.nan) - 1
        feat["ma_long_dev"]     = c / ta.sma(c, MA_LONG).replace(0, np.nan) - 1
        return feat   # 7 features

    def _volume_features(self, df: pd.DataFrame) -> pd.DataFrame:
        feat = pd.DataFrame(index=df.index)
        v   = df["Volume"]
        qv  = df["QuoteVolume"]
        tbr = df["TakerBuyBase"]

        feat["volume_change"]     = v / v.shift(1).replace(0, np.nan) - 1
        feat["volume_ma_ratio"]   = v / ta.sma(v, MA_SHORT).replace(0, np.nan)
        obv = (np.sign(df["Close"].diff()) * v).cumsum()
        feat["obv_ma_ratio"]      = obv / ta.sma(obv, MA_SHORT).replace(0, np.nan)
        feat["quote_vol_ratio"]   = ta.volume_profile_ratio(qv, n_short=24, n_long=MA_SHORT)
        feat["taker_buy_ratio"]   = ta.taker_buy_ratio(tbr, v)
        return feat   # 5 features

    def _technical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        feat = pd.DataFrame(index=df.index)
        c, h, lo = df["Close"], df["High"], df["Low"]

        feat["rsi"]        = ta.rsi(c, RSI_PERIOD)

        macd_l, macd_s, macd_h = ta.macd(c, MACD_FAST, MACD_SLOW, MACD_SIG)
        feat["macd"]       = macd_l
        feat["macd_sig"]   = macd_s
        feat["macd_hist"]  = macd_h

        upper, mid, lower  = ta.bollinger(c, BB_PERIOD, 2.0)
        bw = (upper - lower).replace(0, np.nan)
        feat["bb_width"]   = bw / mid.replace(0, np.nan)
        feat["bb_pos"]     = (c - lower) / bw
        feat["bb_dev"]     = (c - mid) / mid.replace(0, np.nan)

        feat["atr"]        = ta.atr(h, lo, c, ATR_PERIOD) / c.replace(0, np.nan)
        feat["adx"]        = ta.adx(h, lo, c, ADX_PERIOD)

        pk, pd_ = ta.stochastic(h, lo, c, STOCH_K, STOCH_D)
        feat["stoch_k"]    = pk
        feat["stoch_d"]    = pd_

        feat["sma_s_dev"]  = c / ta.sma(c, MA_SHORT).replace(0, np.nan) - 1
        feat["sma_l_dev"]  = c / ta.sma(c, MA_LONG).replace(0, np.nan) - 1
        feat["ema_cross"]  = ta.ema(c, 24) / ta.ema(c, 168).replace(0, np.nan) - 1
        return feat   # 15 features

    def _volatility_features(self, df: pd.DataFrame) -> pd.DataFrame:
        feat = pd.DataFrame(index=df.index)
        ret = np.log(df["Close"] / df["Close"].shift(1))
        h, lo, o, c = df["High"], df["Low"], df["Open"], df["Close"]

        feat["hist_vol_short"] = ta.hist_vol(ret, VOL_SHORT)
        feat["hist_vol_long"]  = ta.hist_vol(ret, VOL_LONG)
        feat["parkinson_vol"]  = ta.parkinson_vol(h, lo, VOL_SHORT)
        feat["gk_vol"]         = ta.garman_klass_vol(h, lo, o, c, VOL_SHORT)
        return feat   # 4 features

    def _crypto_features(self, token: str) -> pd.DataFrame:
        """
        Supplementary + cross-token features.
        Returns 10 features: 5 from supplementary + 1 cross-token + 4 sentiment/onchain.
        """
        supp = self.supp.get(token, pd.DataFrame())
        feat = pd.DataFrame(index=supp.index if not supp.empty else self.ohlcv[token].index)

        def _get(col: str, default: float = 0.0) -> pd.Series:
            if col in supp.columns:
                return supp[col]
            return pd.Series(default, index=feat.index)

        # Funding rate features (3)
        fr = _get("funding_rate")
        feat["funding_rate"]     = fr
        feat["funding_abs"]      = fr.abs()
        feat["funding_momentum"] = ta.funding_rate_momentum(fr, n=3)

        # Market structure (2)
        feat["btc_dominance"]    = _get("btc_dominance_pct")
        feat["vol_mcap_ratio"]   = _get("vol_mcap_ratio")

        # Cross-token BTC correlation (1)
        if self._btc_ret is not None and token != "BTC":
            token_ret = np.log(
                self.ohlcv[token]["Close"] / self.ohlcv[token]["Close"].shift(1)
            )
            # Align index
            btc_aligned = self._btc_ret.reindex(token_ret.index)
            feat["btc_corr"] = ta.rolling_corr_with_btc(token_ret, btc_aligned, CORR_WINDOW)
        else:
            feat["btc_corr"] = 1.0   # BTC vs BTC correlation is always 1

        # Sentiment / public interest (4)
        feat["fear_greed"]        = _get("fear_greed_value") / 100.0   # normalise to [0,1]
        feat["fear_greed_class"]  = _get("fear_greed_class") / 4.0     # normalise to [0,1]
        feat["google_trend"]      = _get(f"trend_{token}") / 100.0
        feat["onchain_pct_chg"]   = _get("onchain_pct_change") / 100.0

        return feat   # 10 features

    # ------------------------------------------------------------------
    # Build API
    # ------------------------------------------------------------------

    def build_token(self, token: str) -> pd.DataFrame:
        """Build the complete 41-feature matrix for one token."""
        df = self.ohlcv[token]

        price_f   = self._price_features(df)        # 7
        volume_f  = self._volume_features(df)        # 5
        tech_f    = self._technical_features(df)     # 15
        vol_f     = self._volatility_features(df)    # 4
        crypto_f  = self._crypto_features(token)     # 10

        combined = pd.concat(
            [price_f, volume_f, tech_f, vol_f, crypto_f], axis=1
        )

        # Align on common index (crypto_f may have been built from supp index)
        combined = combined.reindex(df.index)

        # Drop rows with all-NaN (warm-up period from long rolling windows)
        combined = combined.dropna(how="all")

        # Replace inf
        combined.replace([np.inf, -np.inf], np.nan, inplace=True)

        n_cols = combined.shape[1]
        assert n_cols == N_FEATURES, (
            f"{token}: expected {N_FEATURES} features, got {n_cols}. "
            f"Columns: {list(combined.columns)}"
        )

        logger.info(
            f"{token}: {len(combined):,} rows × {n_cols} features "
            f"| NaN rows: {combined.isna().any(axis=1).sum()}"
        )
        return combined

    def build_all(self) -> Dict[str, pd.DataFrame]:
        for token in self.ohlcv:
            try:
                self.features[token] = self.build_token(token)
            except Exception as exc:
                logger.error(f"{token}: feature build failed — {exc}")
        logger.info(f"Feature matrices built for {len(self.features)} tokens")
        return self.features

    # ------------------------------------------------------------------
    # Sequence creation with rolling z-score normalisation
    # ------------------------------------------------------------------

    def _sequences_for_token(
        self,
        token: str,
        df: pd.DataFrame,
        seq_len: int,
        horizon: int,
        norm_window: int,
    ) -> Tuple[np.ndarray, np.ndarray, List[dict]]:
        """
        Build all sequences for a single token.
        Returns (X_token, y_token, meta_list) without holding other tokens in RAM.
        """
        feat_cols   = list(df.columns)
        n_rows      = len(df)
        n_seqs      = max(0, n_rows - norm_window - seq_len - horizon + 1)

        vals        = df[feat_cols].values.astype(np.float32)
        log_returns = df["log_return"].values

        # Pre-allocate for this token only
        X_tok  = np.empty((n_seqs, seq_len, len(feat_cols)), dtype=np.float32)
        y_tok  = np.empty(n_seqs, dtype=np.float32)
        meta   : List[dict] = []
        count  = 0

        for i in range(norm_window, n_rows - seq_len - horizon + 1):
            target = log_returns[i + seq_len + horizon - 1]
            if not np.isfinite(target):
                continue

            seq    = vals[i : i + seq_len]                   # (seq_len, F)
            window = vals[i - norm_window : i]               # (norm_window, F)

            mu    = np.nanmean(window, axis=0)               # (F,)
            sigma = np.nanstd(window,  axis=0)               # (F,)
            sigma = np.where(sigma < 1e-8, 1.0, sigma)

            seq_norm = (seq - mu) / sigma
            seq_norm = np.where(np.isfinite(seq_norm), seq_norm, 0.0)

            X_tok[count] = seq_norm
            y_tok[count] = target
            meta.append({
                "token":         token,
                "datetime":      df.index[i + seq_len + horizon - 1],
                "target_return": float(target),
            })
            count += 1

        return X_tok[:count], y_tok[:count], meta

    def create_sequences(
        self,
        seq_len: int = SEQUENCE_LENGTH,
        horizon: int = FORECAST_HORIZON,
        norm_window: int = NORM_WINDOW,
        out_dir: "Path | None" = None,
    ) -> Tuple[np.ndarray, np.ndarray, pd.DataFrame]:
        """
        Sliding-window sequences with per-feature rolling z-score normalisation.

        Memory strategy: process one token at a time, write each token's
        sequences to a temporary .npy shard on disk, then memmap-concatenate
        at the end. Peak RAM = max(one token's sequences) ≈ 670 MB instead
        of 5.4 GB for all tokens at once.

        Returns
        -------
        X        : float32  (N, seq_len, N_FEATURES)   memory-mapped
        y        : float32  (N,)
        metadata : DataFrame[token, datetime, target_return]
        """
        import tempfile, os
        from pathlib import Path as _Path

        if out_dir is None:
            from phase1.utils.config import PROCESSED_DIR
            out_dir = PROCESSED_DIR
        out_dir = _Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        shard_paths : List[str]  = []
        all_y       : List[np.ndarray] = []
        all_meta    : List[dict] = []
        total_seqs  = 0

        for token, df in self.features.items():
            n_needed = norm_window + seq_len + horizon
            if len(df) < n_needed:
                logger.warning(f"{token}: only {len(df)} rows, need {n_needed} — skipping")
                continue

            logger.info(f"{token}: building sequences …")
            X_tok, y_tok, meta = self._sequences_for_token(
                token, df, seq_len, horizon, norm_window
            )

            if len(X_tok) == 0:
                logger.warning(f"{token}: 0 valid sequences — skipping")
                continue

            # Write shard to disk immediately, free RAM
            shard = str(out_dir / f"_shard_{token}.npy")
            np.save(shard, X_tok)
            shard_paths.append(shard)
            all_y.append(y_tok)
            all_meta.extend(meta)
            total_seqs += len(X_tok)

            logger.info(f"{token}: {len(X_tok):,} sequences saved to shard")
            del X_tok, y_tok   # explicit free

        if total_seqs == 0:
            raise RuntimeError("No sequences generated — check feature build logs")

        # Concatenate shards into final memmap file
        x_path = str(out_dir / "X_sequences.npy")
        logger.info(f"Concatenating {len(shard_paths)} shards → {x_path}")

        X_out = np.lib.format.open_memmap(
            x_path, mode="w+", dtype=np.float32,
            shape=(total_seqs, seq_len, N_FEATURES)
        )
        cursor = 0
        for shard in shard_paths:
            chunk = np.load(shard, mmap_mode="r")
            n     = len(chunk)
            X_out[cursor : cursor + n] = chunk
            cursor += n
            os.remove(shard)   # delete shard after copying

        y        = np.concatenate(all_y).astype(np.float32)
        metadata = pd.DataFrame(all_meta)

        n_tok = metadata["token"].nunique() if not metadata.empty else 0
        logger.info(
            f"Sequences created — X: {X_out.shape} | y: {y.shape} | tokens: {n_tok}"
        )
        return np.array(X_out), y, metadata
