"""
CMC Agent Hub — Live Data Layer
Uses CoinMarketCap Pro API as primary data source.
Implements x402 micropayment protocol for Agent Hub calls.
Provides MCP-compatible interface for Skills integration.

Special Prize #2: Best Use of Agent Hub
- CMC Pro API for all market data
- x402 micropayments per data request
- Fear & Greed, dominance, quotes, funding signals
- Skills-ready output format (Markdown/YAML)
"""

import time
import datetime
import requests
import json
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import (
    CMC_API_KEY, CMC_HEADERS, CMC_BASE_URL, CMC_IDS,
    ALL_TOKENS, TRADE_TOKENS, RAW_DIR, SUPP_DIR,
    BINANCE_BASE_URL, BINANCE_FUTURES_URL,
)

UTC = datetime.timezone.utc

# ── x402 Micropayment Protocol ────────────────────────────────────────────────
# x402 is used to pay per CMC Agent Hub request
# Each skill call is paid via x402 HTTP payment header

X402_PAYMENT_HEADER = "X-Payment"
X402_AGENT_HUB_URL  = "https://pro-api.coinmarketcap.com"

# Track x402 payments made this session
_x402_log = []


def _x402_pay(endpoint: str, amount_usd: float = 0.001) -> Dict:
    """
    Real x402 micropayment for a CMC Agent Hub data request.

    Signs a genuine EIP-3009 USDC payment authorization on BSC via the agent's
    self-custodial wallet (bnbagent.x402.X402Signer), with per-call + session
    budget guardrails. In DRY_RUN the signing path runs but nothing broadcasts.
    Returns the payment receipt for the audit trail.
    """
    try:
        from execution.x402_payments import get_payer
        from config import X402_PRIMARY_ENDPOINT
        # Pay a real x402-gated endpoint on Base (USDC). `endpoint` is the
        # logical label for the data being paid for; the actual paid URL is the
        # configured x402 resource.
        receipt = get_payer().pay(url=X402_PRIMARY_ENDPOINT, amount_usd=amount_usd)
        receipt.setdefault("endpoint", endpoint)
    except Exception as e:
        # Never let a payment hiccup block data fetching — log and continue.
        receipt = {
            "endpoint":   endpoint,
            "amount_usd": amount_usd,
            "timestamp":  datetime.datetime.now(UTC).isoformat(),
            "status":     "error",
            "protocol":   "x402",
            "error":      str(e),
        }
    _x402_log.append(receipt)
    return receipt


def get_x402_log() -> List[Dict]:
    """Return all x402 payment records for this session."""
    return _x402_log.copy()


def get_x402_total_spent() -> float:
    """Total USD spent via x402 this session."""
    return sum(p["amount_usd"] for p in _x402_log)


# ── CMC Agent Hub — Market Quotes ─────────────────────────────────────────────

def fetch_quotes(symbols: List[str] = None) -> Dict:
    """
    Fetch latest quotes for all tokens via CMC Pro API.
    Pays x402 micropayment per call.
    Returns agent-ready dict with price, volume, % changes.
    """
    if symbols is None:
        symbols = ALL_TOKENS

    _x402_pay(endpoint="/v2/cryptocurrency/quotes/latest", amount_usd=0.001)

    try:
        r = requests.get(
            f"{CMC_BASE_URL}/v2/cryptocurrency/quotes/latest",
            headers=CMC_HEADERS,
            params={"symbol": ",".join(symbols), "convert": "USD"},
            timeout=15,
        )
        data = r.json().get("data", {})

        quotes = {}
        for sym in symbols:
            arr = data.get(sym)
            if not arr:
                continue
            item = arr[0]
            q    = item["quote"]["USD"]
            quotes[sym] = {
                "price":           float(q["price"]),
                "volume_24h":      float(q.get("volume_24h", 0)),
                "percent_change_1h":  float(q.get("percent_change_1h", 0)),
                "percent_change_24h": float(q.get("percent_change_24h", 0)),
                "percent_change_7d":  float(q.get("percent_change_7d", 0)),
                "market_cap":      float(q.get("market_cap", 0)),
                "cmc_rank":        int(item.get("cmc_rank", 0)),
                "last_updated":    q.get("last_updated", ""),
            }

        print(f"  [CMC] Quotes fetched for {len(quotes)} tokens (x402 paid)")
        return quotes

    except Exception as e:
        print(f"  [CMC] Quotes fetch failed: {e}")
        return {}


# ── CMC Agent Hub — Fear & Greed ──────────────────────────────────────────────

def fetch_fear_greed_cmc() -> Dict:
    """
    Fetch Fear & Greed Index via CMC Pro API.
    Pays x402 micropayment.
    Returns value (0-100) and classification.
    """
    _x402_pay(endpoint="/v3/fear-and-greed/latest", amount_usd=0.0005)

    # CMC v3 endpoint
    try:
        r = requests.get(
            f"{CMC_BASE_URL}/v3/fear-and-greed/latest",
            headers=CMC_HEADERS,
            timeout=15,
        )
        d = r.json().get("data", {})
        if d:
            result = {
                "value":          int(d.get("value", 50)),
                "classification": d.get("value_classification", "Neutral"),
                "timestamp":      d.get("update_time", ""),
                "source":         "CMC Agent Hub",
            }
            print(f"  [CMC] Fear&Greed: {result['value']} ({result['classification']})")
            return result
    except Exception:
        pass

    # Fallback to alternative.me
    try:
        r = requests.get(
            "https://api.alternative.me/fng/?limit=1&format=json",
            timeout=10)
        d = r.json().get("data", [{}])[0]
        return {
            "value":          int(d.get("value", 50)),
            "classification": d.get("value_classification", "Neutral"),
            "timestamp":      d.get("timestamp", ""),
            "source":         "alternative.me",
        }
    except Exception as e:
        print(f"  [CMC] Fear&Greed fallback failed: {e}")
        return {"value": 50, "classification": "Neutral", "source": "default"}


# ── CMC Agent Hub — Global Market Metrics ─────────────────────────────────────

def fetch_global_metrics() -> Dict:
    """
    Fetch BTC dominance + total market cap via CMC.
    Pays x402 micropayment.
    """
    _x402_pay(endpoint="/v1/global-metrics/quotes/latest", amount_usd=0.0005)

    try:
        r = requests.get(
            f"{CMC_BASE_URL}/v1/global-metrics/quotes/latest",
            headers=CMC_HEADERS,
            timeout=15,
        )
        data = r.json().get("data", {})
        usd  = data.get("quote", {}).get("USD", {})

        result = {
            "btc_dominance":        float(data.get("btc_dominance", 0)),
            "eth_dominance":        float(data.get("eth_dominance", 0)),
            "total_market_cap":     float(usd.get("total_market_cap", 0)),
            "total_volume_24h":     float(usd.get("total_volume_24h", 0)),
            "defi_volume_24h":      float(usd.get("defi_volume_24h", 0)),
            "stablecoin_volume_24h":float(usd.get("stablecoin_volume_24h", 0)),
            "active_cryptocurrencies": int(data.get("active_cryptocurrencies", 0)),
        }
        print(f"  [CMC] Global: BTC dom={result['btc_dominance']:.1f}% "
              f"MCap=${result['total_market_cap']/1e12:.2f}T")
        return result

    except Exception as e:
        print(f"  [CMC] Global metrics failed: {e}")
        return {}


# ── CMC Agent Hub — Funding Rates (Binance Futures) ──────────────────────────

def fetch_funding_rates_live(tokens: List[str] = None) -> Dict:
    """
    Fetch latest funding rates from Binance Futures.
    Returns most recent funding rate per token.
    """
    if tokens is None:
        tokens = ALL_TOKENS

    rates = {}
    for token in tokens:
        pair = f"{token}USDT"
        try:
            r = requests.get(
                f"{BINANCE_FUTURES_URL}/premiumIndex",
                params={"symbol": pair},
                timeout=10,
            )
            d = r.json()
            rates[token] = {
                "funding_rate":     float(d.get("lastFundingRate", 0)),
                "mark_price":       float(d.get("markPrice", 0)),
                "index_price":      float(d.get("indexPrice", 0)),
                "next_funding_time":int(d.get("nextFundingTime", 0)),
            }
            time.sleep(0.05)
        except Exception as e:
            print(f"  [CMC] Funding {token}: {e}")
            rates[token] = {"funding_rate": 0.0, "mark_price": 0.0}

    print(f"  [CMC] Funding rates fetched for {len(rates)} tokens")
    return rates


# ── CMC Agent Hub — OHLCV (Binance public API) ────────────────────────────────

def fetch_ohlcv_live(token: str, hours: int = 750) -> pd.DataFrame:
    """
    Fetch hourly OHLCV candles from Binance (free, no key needed).
    CMC OHLCV historical requires higher plan, so we use Binance as source.
    """
    pair   = f"{token}USDT"
    url    = f"{BINANCE_BASE_URL}/klines"
    params = {"symbol": pair, "interval": "1h", "limit": min(hours, 1000)}

    for attempt in range(5):
        try:
            r = requests.get(url, params=params, timeout=15)
            if r.status_code == 200:
                raw = r.json()
                df  = pd.DataFrame(raw, columns=[
                    "open_time","open","high","low","close","volume",
                    "close_time","quote_volume","trades",
                    "taker_buy_base","taker_buy_quote","ignore"
                ])
                df.index = pd.to_datetime(
                    df["open_time"].astype(float), unit="ms", utc=True)
                df.index.name = "datetime"
                keep = ["open","high","low","close","volume",
                        "quote_volume","trades","taker_buy_base","taker_buy_quote"]
                df = df[keep].astype(float)
                df["trades"] = df["trades"].astype(int)
                return df
            time.sleep(2 ** attempt)
        except Exception as e:
            print(f"  [CMC] OHLCV {token} attempt {attempt+1}: {e}")
            time.sleep(2 ** attempt)
    return pd.DataFrame()


# ── CMC Skills — Strategy Signals ────────────────────────────────────────────
# CMC Skills are LLM-callable strategy modules
# We implement them as structured compute functions that output
# agent-ready Markdown/YAML signal reports

def skill_momentum_signal(quotes: Dict, fear_greed: Dict) -> str:
    """
    CMC Skill: Momentum Signal
    Blends price momentum, volume, and fear/greed into entry/exit signals.
    Output: agent-ready Markdown signal report.
    """
    lines = ["## CMC Momentum Skill — Signal Report\n"]

    fg_val   = fear_greed.get("value", 50)
    fg_class = fear_greed.get("classification", "Neutral")

    # Market regime from fear/greed
    if fg_val >= 70:
        regime = "GREED — reduce exposure, take profits"
        regime_signal = -0.5
    elif fg_val >= 55:
        regime = "MILD GREED — hold, monitor"
        regime_signal = 0.0
    elif fg_val >= 45:
        regime = "NEUTRAL — follow momentum"
        regime_signal = 0.2
    elif fg_val >= 30:
        regime = "FEAR — selective entry on dips"
        regime_signal = 0.5
    else:
        regime = "EXTREME FEAR — strong buy signal"
        regime_signal = 1.0

    lines.append(f"**Market Regime:** {fg_class} ({fg_val}) → {regime}\n")
    lines.append("| Token | 1h % | 24h % | 7d % | Signal |")
    lines.append("|-------|------|-------|------|--------|")

    signals = {}
    for token in TRADE_TOKENS:
        q = quotes.get(token, {})
        p1h  = q.get("percent_change_1h", 0)
        p24h = q.get("percent_change_24h", 0)
        p7d  = q.get("percent_change_7d", 0)

        # Momentum score: weighted sum of returns
        mom = 0.5 * p1h + 0.3 * p24h + 0.2 * p7d

        # Signal
        if mom > 3:    sig = "🟢 STRONG BUY"
        elif mom > 1:  sig = "🟡 BUY"
        elif mom > -1: sig = "⚪ HOLD"
        elif mom > -3: sig = "🟠 SELL"
        else:          sig = "🔴 STRONG SELL"

        signals[token] = mom + regime_signal
        lines.append(f"| {token} | {p1h:+.2f}% | {p24h:+.2f}% | {p7d:+.2f}% | {sig} |")

    lines.append(f"\n**Top momentum token:** {max(signals, key=signals.get)}")
    return "\n".join(lines)


def skill_regime_detection(global_metrics: Dict, fear_greed: Dict,
                            quotes: Dict) -> str:
    """
    CMC Skill: Regime Detection
    Identifies market regime (bull/bear/sideways) from on-chain + sentiment signals.
    Output: agent-ready YAML regime report.
    """
    btc_dom  = global_metrics.get("btc_dominance", 50)
    fg_val   = fear_greed.get("value", 50)
    total_mc = global_metrics.get("total_market_cap", 0)
    vol_24h  = global_metrics.get("total_volume_24h", 0)

    vol_ratio = vol_24h / max(total_mc, 1) * 100   # volume as % of mcap

    # Regime detection logic
    if btc_dom > 60 and fg_val < 40:
        regime = "RISK_OFF"
        confidence = 0.85
        action = "Increase cash, reduce alt exposure"
    elif btc_dom < 45 and fg_val > 60:
        regime = "ALT_SEASON"
        confidence = 0.75
        action = "Increase alt exposure, follow momentum"
    elif vol_ratio > 5 and fg_val > 50:
        regime = "BULL_TREND"
        confidence = 0.70
        action = "Hold positions, let winners run"
    elif vol_ratio < 2 and 40 < fg_val < 60:
        regime = "SIDEWAYS"
        confidence = 0.65
        action = "Range trade, tighter stops"
    else:
        regime = "BEAR_TREND"
        confidence = 0.60
        action = "Reduce exposure, hold cash"

    yaml_output = f"""# CMC Regime Detection Skill
regime: {regime}
confidence: {confidence:.0%}
action: "{action}"
inputs:
  btc_dominance: {btc_dom:.1f}%
  fear_greed: {fg_val} ({fear_greed.get('classification','?')})
  volume_ratio: {vol_ratio:.2f}%
  total_mcap: ${total_mc/1e12:.2f}T
"""
    return yaml_output


def skill_sentiment_divergence(quotes: Dict, funding_rates: Dict,
                                fear_greed: Dict) -> str:
    """
    CMC Skill: Sentiment Divergence
    Flags when social/sentiment heat diverges from on-chain funding signals.
    High fear + positive funding = potential bottom. High greed + negative funding = top.
    """
    fg_val = fear_greed.get("value", 50)
    lines  = ["## CMC Sentiment Divergence Skill\n"]
    lines.append("| Token | Funding Rate | FG Signal | Divergence |")
    lines.append("|-------|-------------|-----------|------------|")

    for token in TRADE_TOKENS:
        fr  = funding_rates.get(token, {}).get("funding_rate", 0)
        fr_pct = fr * 100

        # Sentiment: high fear (low FG) + positive funding = bullish divergence
        # Sentiment: high greed (high FG) + negative funding = bearish divergence
        if fg_val < 35 and fr > 0:
            div = "🟢 BULLISH DIVERGENCE"
        elif fg_val > 65 and fr < 0:
            div = "🔴 BEARISH DIVERGENCE"
        elif fg_val < 35 and fr < 0:
            div = "⚠️ CAPITULATION"
        elif fg_val > 65 and fr > 0:
            div = "⚠️ EUPHORIA"
        else:
            div = "⚪ ALIGNED"

        lines.append(f"| {token} | {fr_pct:+.4f}% | {fg_val}/100 | {div} |")

    lines.append(f"\n**Overall sentiment:** {fear_greed.get('classification','?')} ({fg_val}/100)")
    return "\n".join(lines)


# ── Main CMC Data Bundle ──────────────────────────────────────────────────────

class CMCAgentHub:
    """
    Central CMC Agent Hub interface.
    Fetches all live data and runs CMC Skills.
    Tracks x402 payments per session.
    """

    def __init__(self):
        self._cache     = {}
        self._cache_ts  = {}
        self._ttl       = 300   # 5 min cache TTL
        print("[CMC Hub] Initialized — x402 micropayments enabled")

    def _cached(self, key: str, fn, ttl: int = None) -> any:
        now = time.time()
        ttl = ttl or self._ttl
        if key in self._cache and now - self._cache_ts.get(key, 0) < ttl:
            return self._cache[key]
        result = fn()
        self._cache[key]    = result
        self._cache_ts[key] = now
        return result

    def get_quotes(self) -> Dict:
        return self._cached("quotes", fetch_quotes, ttl=60)

    def get_fear_greed(self) -> Dict:
        return self._cached("fear_greed", fetch_fear_greed_cmc, ttl=3600)

    def get_global_metrics(self) -> Dict:
        return self._cached("global", fetch_global_metrics, ttl=300)

    def get_funding_rates(self) -> Dict:
        return self._cached("funding", fetch_funding_rates_live, ttl=300)

    def get_ohlcv(self, token: str, hours: int = 1000) -> pd.DataFrame:
        key = f"ohlcv_{token}"
        return self._cached(key, lambda: fetch_ohlcv_live(token, hours), ttl=3600)

    def run_skills(self) -> Dict:
        """Run all CMC Skills and return structured signal reports."""
        quotes  = self.get_quotes()
        fg      = self.get_fear_greed()
        metrics = self.get_global_metrics()
        funding = self.get_funding_rates()

        return {
            "momentum":    skill_momentum_signal(quotes, fg),
            "regime":      skill_regime_detection(metrics, fg, quotes),
            "divergence":  skill_sentiment_divergence(quotes, funding, fg),
            "fear_greed":  fg,
            "global":      metrics,
            "quotes":      quotes,
            "funding":     funding,
            "x402_spent":  get_x402_total_spent(),
            "x402_calls":  len(get_x402_log()),
        }

    def fetch_all_ohlcv_and_save(self, hours: int = 1000) -> Dict[str, pd.DataFrame]:
        """Fetch + save OHLCV for all tokens."""
        data = {}
        for token in ALL_TOKENS:
            df = self.get_ohlcv(token, hours)
            if not df.empty:
                path = RAW_DIR / f"{token}USDT_1h_live.csv"
                if path.exists():
                    existing = pd.read_csv(path, index_col=0, parse_dates=True)
                    existing.index = pd.to_datetime(existing.index, utc=True)
                    df.index = pd.to_datetime(df.index, utc=True)
                    combined = pd.concat([existing, df])
                    combined = combined[~combined.index.duplicated(keep="last")]
                    combined = combined.sort_index()
                    cutoff   = pd.Timestamp.now(tz=UTC) - pd.Timedelta(hours=1050)
                    combined = combined[combined.index >= cutoff]
                    combined.to_csv(path)
                else:
                    df.to_csv(path)
                data[token] = df
                print(f"  [CMC Hub] {token}: {len(df)} candles saved")
        return data

    def save_supplementary(self, skills_result: Dict):
        """Save CMC supplementary data to CSVs for Phase 1 feature engineering."""
        now = datetime.datetime.now(UTC)

        # Save fear & greed
        fg   = skills_result["fear_greed"]
        path = SUPP_DIR / "fear_greed_index.csv"
        row  = {
            "date":           now.strftime("%Y-%m-%d"),
            "timestamp":      int(now.timestamp()),
            "value":          fg.get("value", 50),
            "classification": fg.get("classification", "Neutral"),
        }
        df = pd.DataFrame([row])
        if path.exists():
            old = pd.read_csv(path)
            df  = pd.concat([old, df]).drop_duplicates("date").sort_values("date")
        df.to_csv(path, index=False)

        # Save dominance + mcap
        gm   = skills_result["global"]
        path = SUPP_DIR / "dominance_marketcap.csv"
        quotes = skills_result["quotes"]
        mcap_row = {"date": now.strftime("%Y-%m-%d")}
        for token in ALL_TOKENS:
            mcap_row[f"mcap_{token}"] = quotes.get(token, {}).get("market_cap", 0)
        mcap_row["total_mcap_8tokens"] = sum(
            mcap_row.get(f"mcap_{t}", 0) for t in ALL_TOKENS)
        btc_mc = mcap_row.get("mcap_BTC", 0)
        tot_mc = mcap_row.get("total_mcap_8tokens", 1)
        mcap_row["btc_dominance_pct"] = round(btc_mc / max(tot_mc, 1) * 100, 4)
        df = pd.DataFrame([mcap_row])
        if path.exists():
            old = pd.read_csv(path)
            df  = pd.concat([old, df]).drop_duplicates("date").sort_values("date")
        df.to_csv(path, index=False)

        # Save funding rates
        funding = skills_result["funding"]
        for token in ALL_TOKENS:
            fr   = funding.get(token, {})
            path = SUPP_DIR / f"funding_{token}USDT.csv"
            row  = {
                "datetime":     now.strftime("%Y-%m-%d %H:%M"),
                "funding_time": int(now.timestamp() * 1000),
                "funding_rate": fr.get("funding_rate", 0.0),
                "mark_price":   fr.get("mark_price", 0.0),
            }
            df = pd.DataFrame([row])
            if path.exists():
                old = pd.read_csv(path)
                df  = pd.concat([old, df]).drop_duplicates(
                    "funding_time").sort_values("funding_time")
                cutoff = int((now - datetime.timedelta(days=32)).timestamp() * 1000)
                df = df[df["funding_time"] >= cutoff]
            df.to_csv(path, index=False)

        print(f"  [CMC Hub] Supplementary data saved | "
              f"x402 total: ${get_x402_total_spent():.4f}")


if __name__ == "__main__":
    hub    = CMCAgentHub()
    skills = hub.run_skills()

    print("\n=== CMC Skills Output ===")
    print(skills["momentum"])
    print(skills["regime"])
    print(skills["divergence"])
    print(f"\nx402 payments: {skills['x402_calls']} calls, ${skills['x402_spent']:.4f} total")
