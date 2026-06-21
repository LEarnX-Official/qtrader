"""
Phase 5 data builder.

State  : all 8 tokens (including BTC) → agent sees BTC's Ψ₀/ΣAᵢ/probs as context
Action : 7 trade tokens only (BTC excluded from portfolio weights)
Returns: 7 trade tokens only (used for reward and Kelly fractions)

BTC is the dominant market driver. Including its full quantum state in the
observation gives the agent direct context about market regime, which predicts
alt behaviour far better than indirect signals alone.
"""

import numpy as np
import pandas as pd
from typing import Tuple

from phase1.utils.config import META_CSV, Y_NPY
from phase2.utils.config import LATENTS_ALL
from phase3.utils.config import ALPHA_NPY, SIGMA_AI_NPY
from phase4.utils.config import COLLAPSE_PROBS, NUM_BINS
from phase5.utils.config import (
    ALL_TOKENS, DATA_P5, N_ASSETS, N_STATE_TOKENS,
    TRADE_TOKENS, TRAIN_END, VAL_END,
)
from phase5.utils.logging import get_logger

logger = get_logger(__name__)

_UNIFORM_PROB = 1.0 / NUM_BINS


def load_split_data(split: str = "train") -> Tuple[np.ndarray, ...]:
    """
    Load and pivot arrays for a given split.

    Returns
    -------
    psi0     : (T, 8, 32)   — all 8 tokens including BTC (state context)
    sigma_ai : (T, 8, 512)
    alpha    : (T, 8, 1)
    probs    : (T, 8, 20)
    returns  : (T, 7)       — 7 trade tokens only (no BTC in reward/Kelly)
    """
    psi0_all  = np.load(LATENTS_ALL)
    sigma_all = np.load(SIGMA_AI_NPY)
    alpha_all = np.load(ALPHA_NPY)
    probs_all = np.load(COLLAPSE_PROBS)
    y_all     = np.load(Y_NPY)
    meta      = pd.read_csv(META_CSV, parse_dates=["datetime"])

    # Validate PINN probs
    prob_sums = probs_all.sum(axis=1)
    bad = np.abs(prob_sums - 1.0) > 0.01
    if bad.any():
        logger.warning(f"PINN probs: {bad.sum()} rows don't sum to 1 — re-normalising")
        probs_all = probs_all / (prob_sums[:, None] + 1e-8)

    # Date mask
    if split == "train":
        mask = (meta["datetime"] <= TRAIN_END).values
    else:
        mask = ((meta["datetime"] > TRAIN_END) & (meta["datetime"] <= VAL_END)).values

    sub_meta  = meta[mask].reset_index(drop=True)
    psi0_sub  = psi0_all[mask]
    sigma_sub = sigma_all[mask]
    alpha_sub = alpha_all[mask]
    probs_sub = probs_all[mask]
    y_sub     = y_all[mask]

    # Pivot flat → (T, N_tokens, dim)
    tok_to_idx = {t: i for i, t in enumerate(ALL_TOKENS)}
    datetimes  = np.sort(sub_meta["datetime"].unique())
    T          = len(datetimes)
    dt_to_idx  = {pd.Timestamp(dt): i for i, dt in enumerate(datetimes)}

    # Initialise: probs → uniform (valid), others → zeros
    n_tok    = N_STATE_TOKENS   # 8
    psi0_3d  = np.zeros((T, n_tok, 32),      dtype=np.float32)
    sigma_3d = np.zeros((T, n_tok, 512),     dtype=np.float32)
    alpha_3d = np.zeros((T, n_tok, 1),       dtype=np.float32)
    probs_3d = np.full((T, n_tok, NUM_BINS), _UNIFORM_PROB, dtype=np.float32)
    y_3d     = np.zeros((T, n_tok),          dtype=np.float32)

    missing = 0
    for row_i, row in sub_meta.iterrows():
        tok = row["token"]
        if tok not in tok_to_idx:
            missing += 1
            continue
        t_idx = dt_to_idx[pd.Timestamp(row["datetime"])]
        k     = tok_to_idx[tok]
        psi0_3d [t_idx, k]   = psi0_sub [row_i]
        sigma_3d[t_idx, k]   = sigma_sub[row_i]
        alpha_3d[t_idx, k]   = alpha_sub[row_i]
        probs_3d[t_idx, k]   = probs_sub[row_i]
        y_3d    [t_idx, k]   = y_sub    [row_i]

    if missing:
        logger.warning(f"Split '{split}': {missing} rows had unknown token — skipped")

    # Guard: any remaining near-zero probs → uniform
    bad_mask = probs_3d.sum(axis=2) < 0.5
    if bad_mask.any():
        logger.warning(f"Split '{split}': {bad_mask.sum()} cells forced to uniform probs")
        probs_3d[bad_mask] = _UNIFORM_PROB

    # ── State arrays: ALL 8 tokens (agent observes BTC for context) ──
    psi0_out  = psi0_3d   # (T, 8, 32)
    sigma_out = sigma_3d  # (T, 8, 512)
    alpha_out = alpha_3d  # (T, 8, 1)
    probs_out = probs_3d  # (T, 8, 20)

    # ── Return array: 7 TRADE tokens only (no BTC in reward/Kelly) ───
    trade_idx = [tok_to_idx[t] for t in TRADE_TOKENS]
    y_out     = y_3d[:, trade_idx]   # (T, 7)

    out_sums = probs_out.sum(axis=2)
    logger.info(
        f"Split '{split}': T={T} | "
        f"state=(T,8,565) trade_returns=(T,7) | "
        f"prob_sum min={out_sums.min():.4f} max={out_sums.max():.4f}"
    )

    # Cache
    DATA_P5.mkdir(parents=True, exist_ok=True)
    np.save(DATA_P5 / f"psi0_{split}.npy",   psi0_out)
    np.save(DATA_P5 / f"sigma_{split}.npy",  sigma_out)
    np.save(DATA_P5 / f"alpha_{split}.npy",  alpha_out)
    np.save(DATA_P5 / f"probs_{split}.npy",  probs_out)
    np.save(DATA_P5 / f"y_{split}.npy",      y_out)

    return psi0_out, sigma_out, alpha_out, probs_out, y_out


def load_cached_split(split: str) -> Tuple[np.ndarray, ...]:
    needed = [DATA_P5 / f"{k}_{split}.npy"
              for k in ("psi0", "sigma", "alpha", "probs", "y")]
    if all(p.exists() for p in needed):
        logger.info(f"Loading cached split '{split}'")
        return tuple(np.load(str(p)) for p in needed)
    return load_split_data(split)
