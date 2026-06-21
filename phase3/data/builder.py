"""
Phase 3 multimodal feature builder.

Constructs 256-dim observer vectors from:
  - VAE latent (32)        — Ψ₀ from Phase 2
  - Funding rate stats (16) — funding_rate, abs, momentum per token
  - Fear/Greed (16)         — value (norm), class (norm) × repeated
  - On-chain proxy (32)     — vol_mcap_ratio, pct_chg, trades, own_mcap
  - BTC dominance (32)      — btc_dominance_pct, total_mcap repeated
  - Market/Technical (96)   — rolling mean/std of price+vol+tech features
  - Macro/Corr (32)         — btc_corr, google_trend, cross-token signals

All features are already z-score normalised by Phase 1's rolling window,
so no additional normalisation is needed here.
"""

import numpy as np
import pandas as pd
from typing import Dict, Tuple

from phase3.utils.config import INPUT_DIM, MODALITY_DIMS
from phase3.utils.logging import get_logger

logger = get_logger(__name__)


def build_multimodal_features(
    latents:  np.ndarray,     # (N, 32)   — Phase 2 VAE latents
    raw_feat: np.ndarray,     # (N, 168, 40) — Phase 1 X sequences (mmap OK)
    supp_df:  pd.DataFrame,   # (N,) rows indexed by sequence number,
                              #   columns from supplementary loader
) -> np.ndarray:
    """
    Assembles the 256-dim multimodal feature vector for each of the N sequences.

    Strategy: for each sequence we take the LAST time-step's features
    (most recent bar in the 168h window) as the point-in-time observer signal,
    plus the VAE latent for the full window.

    Returns
    -------
    mm : (N, 256) float32
    """
    N = len(latents)
    mm = np.zeros((N, INPUT_DIM), dtype=np.float32)

    # Last time-step features from Phase 1: shape (N, 40)
    last_step = raw_feat[:, -1, :]   # (N, 40)

    # Feature index map from Phase 1 (same order as engineer.py)
    # Price(7) Volume(5) Technical(15) Volatility(4) Crypto(10) = 40
    # Crypto block starts at index 31:
    #   31: funding_rate, 32: funding_abs, 33: funding_momentum
    #   34: btc_dominance, 35: vol_mcap_ratio, 36: btc_corr
    #   37: fear_greed,  38: fear_greed_class, 39: google_trend, 40: onchain_pct_chg
    # (0-indexed: 30..39)
    PRICE_SLICE  = slice(0,  7)
    VOL_SLICE    = slice(7,  12)
    TECH_SLICE   = slice(12, 27)
    VOLAT_SLICE  = slice(27, 31)
    FUND_IDX     = [30, 31, 32]      # funding_rate, funding_abs, funding_momentum
    DOM_IDX      = [33]              # btc_dominance
    VMCAP_IDX    = [34]              # vol_mcap_ratio
    CORR_IDX     = [35]              # btc_corr
    FGVAL_IDX    = [36]              # fear_greed (norm)
    FGCLS_IDX    = [37]              # fear_greed_class (norm)
    GTRND_IDX    = [38]              # google_trend
    ONCH_IDX     = [39]              # onchain_pct_chg

    cursor = 0

    # 1. psi0_latent (32)
    d = MODALITY_DIMS["psi0_latent"]
    mm[:, cursor:cursor+d] = latents[:, :d]
    cursor += d

    # 2. funding (16) — funding_rate × 3 features repeated to fill 16
    d = MODALITY_DIMS["funding"]
    fund = last_step[:, FUND_IDX]              # (N, 3)
    tile = int(np.ceil(d / fund.shape[1]))
    mm[:, cursor:cursor+d] = np.tile(fund, (1, tile))[:, :d]
    cursor += d

    # 3. fear_greed (16)
    d = MODALITY_DIMS["fear_greed"]
    fg = last_step[:, FGVAL_IDX + FGCLS_IDX]  # (N, 2)
    tile = int(np.ceil(d / fg.shape[1]))
    mm[:, cursor:cursor+d] = np.tile(fg, (1, tile))[:, :d]
    cursor += d

    # 4. onchain (32) — vol_mcap, pct_chg, onchain_trades (approx from raw), own_mcap
    d = MODALITY_DIMS["onchain"]
    oc = last_step[:, VMCAP_IDX + ONCH_IDX]   # (N, 2)
    tile = int(np.ceil(d / oc.shape[1]))
    mm[:, cursor:cursor+d] = np.tile(oc, (1, tile))[:, :d]
    cursor += d

    # 5. dominance (32)
    d = MODALITY_DIMS["dominance"]
    dom = last_step[:, DOM_IDX]                # (N, 1)
    tile = int(np.ceil(d / dom.shape[1]))
    mm[:, cursor:cursor+d] = np.tile(dom, (1, tile))[:, :d]
    cursor += d

    # 6. market_tech (96) — price + vol + tech + volatility features
    d = MODALITY_DIMS["market_tech"]
    mkt = last_step[:, list(range(0,31))]      # (N, 31)
    tile = int(np.ceil(d / mkt.shape[1]))
    mm[:, cursor:cursor+d] = np.tile(mkt, (1, tile))[:, :d]
    cursor += d

    # 7. macro_corr (32) — btc_corr + google_trend
    d = MODALITY_DIMS["macro_corr"]
    macro = last_step[:, CORR_IDX + GTRND_IDX] # (N, 2)
    tile = int(np.ceil(d / macro.shape[1]))
    mm[:, cursor:cursor+d] = np.tile(macro, (1, tile))[:, :d]
    cursor += d

    assert cursor == INPUT_DIM, f"cursor={cursor} != INPUT_DIM={INPUT_DIM}"

    # Replace any nan/inf
    mm = np.where(np.isfinite(mm), mm, 0.0)
    logger.info(f"Multimodal features built: {mm.shape}")
    return mm
