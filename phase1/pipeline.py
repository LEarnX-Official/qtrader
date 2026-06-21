"""
Phase 1 Pipeline — Quantum Trader

Orchestrates:
    1. Load Binance 1h OHLCV for 8 crypto tokens
    2. Load & forward-fill all 5 supplementary sources
    3. Engineer 41 features per token
    4. Build sliding-window sequences (seq_len=168, horizon=4)
    5. Save X.npy, y.npy, metadata.csv

Usage
-----
    from phase1.pipeline import Phase1Pipeline
    result = Phase1Pipeline().run()
    # result.X, result.y, result.metadata, result.ohlcv, result.supp
"""

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd

from phase1.loaders.market import CryptoDataLoader
from phase1.loaders.supplementary import SupplementaryLoader
from phase1.features.engineer import FeatureEngineer, N_FEATURES  # noqa: F401
from phase1.utils.config import (
    FORECAST_HORIZON, META_CSV, NORM_WINDOW,
    PROCESSED_DIR, SEQUENCE_LENGTH, TOKENS,
    X_NPY, Y_NPY,
)
from phase1.utils.logging import get_logger, setup_logging

logger = get_logger(__name__)


@dataclass
class Phase1Result:
    """All Phase 1 outputs consumed by Phase 2 (VAE)."""
    X:        np.ndarray              # (N, seq_len, N_FEATURES) float32
    y:        np.ndarray              # (N,) float32
    metadata: pd.DataFrame            # token, datetime, target_return
    ohlcv:    Dict[str, pd.DataFrame] # raw + cleaned OHLCV
    supp:     Dict[str, pd.DataFrame] # hourly-aligned supplementary data


class Phase1Pipeline:
    """
    Full Phase 1 data pipeline for Quantum Trader.

    Parameters
    ----------
    sequence_length  : look-back window in hours (default 168 = 1 week)
    forecast_horizon : bars ahead to predict (default 4h)
    norm_window      : rolling z-score window (default 720h = 30d)
    force_rebuild    : ignore cached artefacts and rebuild from scratch
    """

    def __init__(
        self,
        sequence_length:  int  = SEQUENCE_LENGTH,
        forecast_horizon: int  = FORECAST_HORIZON,
        norm_window:      int  = NORM_WINDOW,
        force_rebuild:    bool = False,
    ) -> None:
        self.seq_len   = sequence_length
        self.horizon   = forecast_horizon
        self.norm_win  = norm_window
        self.force     = force_rebuild

    # ------------------------------------------------------------------
    # Steps
    # ------------------------------------------------------------------

    def step1_load_ohlcv(self) -> Dict[str, pd.DataFrame]:
        """Load and validate all 8 token OHLCV CSVs."""
        loader = CryptoDataLoader()
        data   = loader.load_all()

        report = loader.quality_report(data)
        bad    = report[~report["ok"]]
        if not bad.empty:
            logger.warning(f"Quality issues:\n{bad}")

        missing = set(TOKENS) - set(data.keys())
        if missing:
            logger.error(f"Missing tokens: {missing}")
            sys.exit(1)

        return data

    def step2_load_supplementary(
        self, ohlcv: Dict[str, pd.DataFrame]
    ) -> Dict[str, pd.DataFrame]:
        """Load all supplementary sources, forward-fill to hourly index."""
        # Use BTC index as the reference hourly index (all tokens share it)
        hourly_idx = ohlcv["BTC"].index
        loader     = SupplementaryLoader(hourly_idx)
        return loader.load_all()

    def step3_features_and_sequences(
        self,
        ohlcv: Dict[str, pd.DataFrame],
        supp:  Dict[str, pd.DataFrame],
    ) -> "Phase1Result":
        """Build feature matrices and sliding-window sequences."""
        eng = FeatureEngineer(ohlcv, supp)
        eng.build_all()

        X, y, metadata = eng.create_sequences(
            seq_len=self.seq_len,
            horizon=self.horizon,
            norm_window=self.norm_win,
            out_dir=PROCESSED_DIR,
        )
        return Phase1Result(X=X, y=y, metadata=metadata, ohlcv=ohlcv, supp=supp)

    # ------------------------------------------------------------------
    # Full run
    # ------------------------------------------------------------------

    def run(self) -> "Phase1Result":
        setup_logging()

        # Fast path: load cached artefacts if they exist
        if not self.force and X_NPY.exists() and Y_NPY.exists() and META_CSV.exists():
            logger.info("Cached artefacts found — loading (use force_rebuild=True to redo)")
            return self._load_cached()

        _banner("STEP 1/3  ─  Load OHLCV")
        ohlcv = self.step1_load_ohlcv()

        _banner("STEP 2/3  ─  Load Supplementary Data")
        supp  = self.step2_load_supplementary(ohlcv)

        _banner("STEP 3/3  ─  Feature Engineering & Sequences")
        result = self.step3_features_and_sequences(ohlcv, supp)

        _banner("SAVING ARTEFACTS")
        self._save(result)

        _banner("PHASE 1 CHECKLIST")
        self._checklist(result)

        return result

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def _save(self, result: Phase1Result) -> None:
        PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
        # X_NPY is already written by create_sequences (memmap strategy).
        # Only save y and metadata here.
        np.save(Y_NPY, result.y)
        result.metadata.to_csv(META_CSV, index=False)
        logger.info(f"X        → {X_NPY}   {result.X.shape}")
        logger.info(f"y        → {Y_NPY}   {result.y.shape}")
        logger.info(f"metadata → {META_CSV}  {len(result.metadata):,} rows")

    def _load_cached(self) -> "Phase1Result":
        X        = np.load(X_NPY)
        y        = np.load(Y_NPY)
        metadata = pd.read_csv(META_CSV, parse_dates=["datetime"])
        logger.info(f"Loaded cached X: {X.shape}")
        return Phase1Result(X=X, y=y, metadata=metadata, ohlcv={}, supp={})

    @staticmethod
    def load_result() -> Optional["Phase1Result"]:
        """Load Phase 1 artefacts for use by Phase 2+. Returns None if missing."""
        if not (X_NPY.exists() and Y_NPY.exists() and META_CSV.exists()):
            logger.warning("Phase 1 artefacts not found — run Phase1Pipeline().run() first")
            return None
        X        = np.load(X_NPY)
        y        = np.load(Y_NPY)
        metadata = pd.read_csv(META_CSV, parse_dates=["datetime"])
        logger.info(f"Phase 1 artefacts loaded — X: {X.shape}")
        return Phase1Result(X=X, y=y, metadata=metadata, ohlcv={}, supp={})

    # ------------------------------------------------------------------
    # Checklist
    # ------------------------------------------------------------------

    @staticmethod
    def _checklist(result: Phase1Result) -> None:
        X, y = result.X, result.y
        n_tokens = result.metadata["token"].nunique() if not result.metadata.empty else 0

        checks = {
            f"All 8 tokens loaded ({n_tokens}/8)":              n_tokens == 8,
            f"Sequences N > 100,000  (N={len(X):,})":           len(X) > 100_000,
            f"X shape (N, {SEQUENCE_LENGTH}, {N_FEATURES})":    X.shape[1:] == (SEQUENCE_LENGTH, N_FEATURES),
            "No NaN in X":                                       bool(np.isfinite(X).all()),
            "No NaN in y":                                       bool(np.isfinite(y).all()),
            f"X_sequences.npy saved":                            X_NPY.exists(),
            f"y_targets.npy saved":                              Y_NPY.exists(),
            f"metadata.csv saved":                               META_CSV.exists(),
        }

        all_ok = True
        for label, passed in checks.items():
            icon = "✓" if passed else "✗"
            logger.info(f"  [{icon}] {label}")
            if not passed:
                all_ok = False

        if all_ok:
            logger.info("Phase 1 COMPLETE — ready for Phase 2 (VAE / Ψ₀ estimation)")
        else:
            logger.warning("Phase 1 finished with failures — review logs above")


# ---------------------------------------------------------------------------

def _banner(title: str) -> None:
    logger.info("=" * 60)
    logger.info(f"  {title}")
    logger.info("=" * 60)
