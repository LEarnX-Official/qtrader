"""
CMC Strategy Skill — Regime-Gated Conviction Allocator (reference implementation)

BNB Hack Track 2 — Strategy Skills.

Pure, self-contained strategy function: given CMC Agent Hub data for one point
in time, return target portfolio weights + cash. No execution, no side effects —
exactly the "backtestable strategy spec" Track 2 asks for.

See SKILL.md for the full rule set and BACKTEST.md for evaluation.

Usage
-----
    from skill_strategy import decide_allocation

    weights, cash = decide_allocation(
        fear_greed=42,
        btc_dominance_change=-0.3,
        quotes={"ETH": {"p1h": 0.4, "p24h": 2.1, "p7d": 5.0},
                "XRP": {"p1h": 1.2, "p24h": 3.0, "p7d": 8.0}, ...},
        funding={"ETH": 0.01, "XRP": 0.05, ...},
    )
"""

from __future__ import annotations
from typing import Dict, Tuple

UNIVERSE = ["BNB", "SOL", "ETH", "XRP", "INJ", "DOGE", "LTC"]
MIN_WEIGHT = 0.05   # positions below 5% fold into cash


# ── Step 1: regime gate → risk budget ─────────────────────────────────────────
def risk_budget(fear_greed: float, btc_dominance_change: float = 0.0) -> float:
    """Fear & Greed → fraction of portfolio allowed to be deployed [0,1]."""
    fg = fear_greed
    if   fg < 20:  R = 1.00   # extreme fear → contrarian max deploy
    elif fg < 35:  R = 0.70
    elif fg < 55:  R = 0.50
    elif fg < 70:  R = 0.30
    else:          R = 0.10   # extreme greed → de-risk

    # Alt risk-off: BTC dominance rising fast → trim deployable
    if btc_dominance_change > 0.5:
        R *= 0.8
    return max(0.0, min(1.0, R))


# ── Step 2: momentum conviction ───────────────────────────────────────────────
def momentum_score(p1h: float, p24h: float, p7d: float) -> float:
    """Blended multi-horizon momentum; long-only (negatives → 0)."""
    mom = 0.5 * p1h + 0.3 * p24h + 0.2 * p7d
    return max(0.0, mom)


# ── Step 3: sentiment-divergence filter ───────────────────────────────────────
def divergence_adjust(score: float, funding: float, p24h: float) -> float:
    """Funding vs price → fade crowded longs, lean into capitulation."""
    if funding > 0.03 and p24h > 3:        # crowded long / euphoria
        return score * 0.5
    if funding < -0.01 and p24h < 0:       # capitulation → contrarian entry
        return score * 1.25
    return score


# ── Step 4+5: allocate ────────────────────────────────────────────────────────
def decide_allocation(
    fear_greed: float,
    quotes: Dict[str, Dict[str, float]],
    funding: Dict[str, float] | None = None,
    btc_dominance_change: float = 0.0,
) -> Tuple[Dict[str, float], float]:
    """
    Returns (weights_by_token, cash_weight). Weights + cash sum to 1.0.
    """
    funding = funding or {}
    R = risk_budget(fear_greed, btc_dominance_change)

    raw: Dict[str, float] = {}
    for tok in UNIVERSE:
        q = quotes.get(tok, {})
        s = momentum_score(q.get("p1h", 0.0), q.get("p24h", 0.0), q.get("p7d", 0.0))
        s = divergence_adjust(s, funding.get(tok, 0.0), q.get("p24h", 0.0))
        raw[tok] = s

    total = sum(raw.values())
    weights: Dict[str, float] = {t: 0.0 for t in UNIVERSE}

    if total > 0:
        for tok in UNIVERSE:
            w = (raw[tok] / total) * R          # normalize within risk budget
            weights[tok] = w if w >= MIN_WEIGHT else 0.0
        # renormalize after zeroing dust, still capped by R
        kept = sum(weights.values())
        if kept > 0:
            for tok in UNIVERSE:
                weights[tok] = weights[tok] / kept * min(R, 1.0)

    cash = max(0.0, 1.0 - sum(weights.values()))
    return weights, cash


# ── Demo ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    # Example: neutral-fear regime, ETH/XRP trending, DOGE crowded long
    quotes = {
        "ETH":  {"p1h": 0.4, "p24h": 2.1, "p7d": 5.0},
        "XRP":  {"p1h": 1.2, "p24h": 3.0, "p7d": 8.0},
        "INJ":  {"p1h": 0.1, "p24h": 1.0, "p7d": 2.0},
        "DOGE": {"p1h": 2.0, "p24h": 6.0, "p7d": 12.0},
        "LTC":  {"p1h": -0.5, "p24h": -1.0, "p7d": -2.0},
        "BNB":  {"p1h": 0.2, "p24h": 0.5, "p7d": 1.0},
        "SOL":  {"p1h": 0.3, "p24h": 1.5, "p7d": 4.0},
    }
    funding = {"DOGE": 0.05}   # DOGE crowded → faded
    w, cash = decide_allocation(fear_greed=42, quotes=quotes, funding=funding)
    print("Fear&Greed=42 (neutral) → risk budget applied\n")
    for t, val in sorted(w.items(), key=lambda x: -x[1]):
        if val > 0:
            print(f"  {t:5} {val:6.1%}")
    print(f"  {'CASH':5} {cash:6.1%}")
    print(f"\n  sum = {sum(w.values()) + cash:.3f}")
