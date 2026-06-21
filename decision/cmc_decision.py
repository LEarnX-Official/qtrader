"""
CMC-Skill Decision Layer
========================
Drives allocation for the 8 tokens using CoinMarketCap Agent Hub skills
as the decision engine (not a price-prediction ML model — our research
proved direction is unpredictable from OHLCV; these skills add genuinely
different data: derivatives positioning, CVD, liquidation maps, regime).

Design (honest, defensive, competition-ready):
  1. detect_market_regime  → overall risk budget (cash-heavy when uncertain)
  2. perp_contract_analysis per token → per-token conviction (bias + flow)
  3. Allocate ONLY to tokens with confirmed bullish positioning;
     stay in cash when regime is fearful / signals are neutral.
  4. Output weights → consumed by the TWAK execution layer.

This wins "Best Use of Agent Hub" by using the skills deeply, and survives
the 30% drawdown cap by being defensive when conviction is low.

The skills are called via the cmc-skill-hub MCP tools. Since MCP calls
happen at the orchestration layer (the agent runtime), this module defines
the DECISION LOGIC that consumes skill outputs. The agent passes skill
results in; this module returns target weights.
"""

import datetime
from typing import Dict, List, Optional

UTC = datetime.timezone.utc

# Competition-eligible tokens (BNB/SOL excluded — not on the 149 list)
ELIGIBLE = ["ETH", "XRP", "INJ", "DOGE", "LTC"]
ALL_TOKENS = ["BTC", "BNB", "SOL", "ETH", "XRP", "INJ", "DOGE", "LTC"]


def interpret_regime(regime_result: Dict) -> Dict:
    """
    Parse detect_market_regime output → overall risk budget (0..1).
    Defensive: high risk budget only when regime is constructive.
    """
    data = regime_result.get("data", regime_result)
    report = data.get("report", {})
    regime = report.get("market_regime", "unknown")
    conviction = report.get("conviction", "low")
    metrics = report.get("metrics", {})
    fng = metrics.get("fear_greed_value", 50)

    # Risk budget = fraction of capital allowed to be deployed
    budget = 0.0
    if regime in ("trend_expansion",):
        budget = 0.9 if conviction == "high" else 0.6
    elif regime in ("range_chop", "mixed_transition"):
        budget = 0.3 if conviction != "low" else 0.15
    elif regime in ("overheated_longs", "liquidation_stress"):
        budget = 0.10   # risk-off
    else:
        budget = 0.20

    # Extra defense in extreme fear (but fear can precede bounces — keep some)
    if fng <= 20:
        budget = min(budget, 0.30)   # cap exposure but don't go fully flat

    return {
        "regime": regime,
        "conviction": conviction,
        "fear_greed": fng,
        "risk_budget": round(budget, 3),
    }


def interpret_perp(perp_result: Dict) -> Dict:
    """
    Parse perp_contract_analysis output → per-token conviction score (-1..+1).
    Positive = bullish positioning confirmed, negative = bearish, 0 = neutral.
    """
    data = perp_result.get("data", perp_result)
    ag = data.get("action_guidance", {})
    bias = ag.get("bias", "neutral")
    mdc = ag.get("market_direction_context", {})
    direction = mdc.get("direction", "none")
    status = mdc.get("status", "mixed")

    # Conviction score
    score = 0.0
    if bias == "bullish" and direction in ("up", "long"):
        score = 0.8 if status == "aligned" else 0.4
    elif bias == "bearish" and direction in ("down", "short"):
        score = -0.8 if status == "aligned" else -0.4
    else:
        score = 0.0   # neutral / unconfirmed → no position

    return {
        "bias": bias,
        "direction": direction,
        "status": status,
        "conviction": round(score, 3),
    }


def decide_weights(
    regime_result: Dict,
    perp_results: Dict[str, Dict],
    max_position: float = 0.40,
) -> Dict:
    """
    Combine regime risk-budget + per-token perp conviction → target weights.

    Args:
        regime_result: output of detect_market_regime
        perp_results: {token: perp_contract_analysis output} for eligible tokens
    Returns:
        {weights: {token: w}, cash: float, rationale: str, details: {...}}
    """
    regime = interpret_regime(regime_result)
    budget = regime["risk_budget"]

    # Score each eligible token
    convictions = {}
    for tok in ELIGIBLE:
        if tok in perp_results:
            convictions[tok] = interpret_perp(perp_results[tok])
        else:
            convictions[tok] = {"conviction": 0.0, "bias": "neutral",
                                "direction": "none", "status": "missing"}

    # Only LONG positions (spot-only on BSC). Positive conviction → weight.
    pos_scores = {t: max(0.0, c["conviction"]) for t, c in convictions.items()}
    total_score = sum(pos_scores.values())

    weights = {t: 0.0 for t in ELIGIBLE}
    if total_score > 1e-6 and budget > 0:
        # distribute risk budget proportional to conviction, capped per token
        for t in ELIGIBLE:
            w = budget * (pos_scores[t] / total_score)
            weights[t] = min(w, max_position)
        # renormalize if capping changed the sum
        wsum = sum(weights.values())
        if wsum > budget:
            for t in ELIGIBLE:
                weights[t] *= budget / wsum

    cash = round(1.0 - sum(weights.values()), 4)

    # Rationale
    n_long = sum(1 for w in weights.values() if w > 0.01)
    rationale = (
        f"Regime={regime['regime']}({regime['conviction']}) "
        f"FnG={regime['fear_greed']} → risk_budget={budget:.0%}. "
        f"{n_long} token(s) with confirmed bullish positioning. "
        f"Cash={cash:.0%}."
    )

    return {
        "weights": {t: round(w, 4) for t, w in weights.items()},
        "cash": cash,
        "rationale": rationale,
        "regime": regime,
        "convictions": convictions,
        "timestamp": datetime.datetime.now(UTC).isoformat(),
    }


if __name__ == "__main__":
    # Demo with the live data we fetched
    demo_regime = {
        "data": {"report": {
            "market_regime": "mixed_transition", "conviction": "low",
            "metrics": {"fear_greed_value": 15.0}
        }}
    }
    demo_perp = {
        "ETH": {"data": {"action_guidance": {"bias": "neutral",
                "market_direction_context": {"direction": "none", "status": "mixed"}}}},
    }
    result = decide_weights(demo_regime, {"ETH": demo_perp["ETH"]})
    print("Decision:")
    print(f"  {result['rationale']}")
    print(f"  Weights: {result['weights']}")
    print(f"  Cash: {result['cash']:.0%}")
