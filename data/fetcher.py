"""
CMC Live Data Fetcher
Fetches latest 168h OHLCV + all supplementary data every 1h cycle.
Uses CMC Pro API as primary, Binance public API as fallback for OHLCV.
"""

import time
import datetime
import requests
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Optional

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import (
    CMC_API_KEY, CMC_HEADERS, CMC_BASE_URL, CMC_IDS,
    BINANCE_BASE_URL, BINANCE_FUTURES_URL,
    ALL_TOKENS, RAW_DIR, SUPP_DIR, SEQUENCE_LENGTH,
)

UTC = datetime.timezone.utc


def _ts_now_ms() -> int:
    return int(datetime.datetime.now(UTC).timestamp() * 1000)


def _from_ms(ms: int) -> datetime.datetime:
    return datetime.datetime.fromtimestamp(ms / 1000, tz=UTC)


# ══════════════════════════════════════════════════════════════════════════════
# OHLCV — Binance 1h candles (free, no key needed)
# ══════════════════════════════════════════════════════════════════════════════

def fetch_ohlcv_binance(token: str, hours: int = 200) -> pd.DataFrame:
    """
    Fetch last `hours` hourly candles from Binance for a token.
    Returns DataFrame with columns: open, high, low, close, volume,
    quote_volume, trades, taker_buy_base, taker_buy_quote
    indexed by UTC datetime.
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
                    "open_time", "open", "high", "low", "close", "volume",
                    "close_time", "quote_volume", "trades",
                    "taker_buy_base", "taker_buy_quote", "ignore"
                ])
                df.index = pd.to_datetime(
                    df["open_time"].astype(float), unit="ms", utc=True)
                df.index.name = "datetime"
                keep = ["open","high","low","close","volume",
                        "quote_volume","trades","taker_buy_base","taker_buy_quote"]
                df = df[keep].astype(float)
                df["trades"] = df["trades"].astype(int)
                return df
            elif r.status_code == 429:
                time.sleep(30)
            else:
                time.sleep(2 ** attempt)
        except Exception as e:
            print(f"  Binance OHLCV {token} attempt {attempt+1}: {e}")
            time.sleep(2 ** attempt)
    return pd.DataFrame()


def fetch_all_ohlcv(hours: int = 200) -> Dict[str, pd.DataFrame]:
    """Fetch OHLCV for all 8 tokens and save to RAW_DIR."""
    data = {}
    for token in ALL_TOKENS:
        df = fetch_ohlcv_binance(token, hours)
        if not df.empty:
            path = RAW_DIR / f"{token}USDT_1h_live.csv"
            # Append to existing file or create new
            if path.exists():
                existing = pd.read_csv(path, index_col=0, parse_dates=True)
                existing.index = pd.to_datetime(existing.index, utc=True)
                combined = pd.concat([existing, df])
                combined = combined[~combined.index.duplicated(keep="last")]
                combined = combined.sort_index()
                # Keep only last 30 days (720h) to save disk
                cutoff = pd.Timestamp.now(tz=UTC) - pd.Timedelta(hours=750)
                combined = combined[combined.index >= cutoff]
                combined.to_csv(path)
            else:
                df.to_csv(path)
            data[token] = df
            print(f"  {token}: {len(df)} candles fetched")
        else:
            print(f"  {token}: FAILED to fetch OHLCV")
    return data


# ══════════════════════════════════════════════════════════════════════════════
# FEAR & GREED INDEX
# ══════════════════════════════════════════════════════════════════════════════

def fetch_fear_greed() -> pd.DataFrame:
    """Fetch latest Fear & Greed data from alternative.me."""
    try:
        r = requests.get(
            "https://api.alternative.me/fng/?limit=30&format=json", timeout=15)
        data = r.json().get("data", [])
        rows = []
        for d in data:
            ts = int(d["timestamp"])
            dt = datetime.datetime.fromtimestamp(ts, tz=UTC)
            rows.append({
                "date":           dt.strftime("%Y-%m-%d"),
                "timestamp":      ts,
                "value":          int(d["value"]),
                "classification": d["value_classification"],
            })
        rows.sort(key=lambda x: x["timestamp"])
        df = pd.DataFrame(rows)
        path = SUPP_DIR / "fear_greed_index.csv"
        if path.exists():
            old = pd.read_csv(path)
            df  = pd.concat([old, df]).drop_duplicates("date").sort_values("date")
        df.to_csv(path, index=False)
        print(f"  Fear/Greed: {len(df)} rows")
        return df
    except Exception as e:
        print(f"  Fear/Greed fetch failed: {e}")
        return pd.DataFrame()


# ══════════════════════════════════════════════════════════════════════════════
# FUNDING RATES — Binance Futures
# ══════════════════════════════════════════════════════════════════════════════

def fetch_funding_rates(lookback_hours: int = 200) -> Dict[str, pd.DataFrame]:
    """Fetch recent funding rates for all tokens."""
    base   = BINANCE_FUTURES_URL + "/fundingRate"
    start  = _ts_now_ms() - lookback_hours * 3600 * 1000
    result = {}

    for token in ALL_TOKENS:
        pair = f"{token}USDT"
        try:
            r = requests.get(base, params={
                "symbol": pair, "startTime": start, "limit": 1000
            }, timeout=15)
            data = r.json()
            if not isinstance(data, list):
                continue
            rows = []
            for row in data:
                ft = int(row["fundingTime"])
                dt = _from_ms(ft).strftime("%Y-%m-%d %H:%M")
                rows.append({
                    "datetime":     dt,
                    "funding_time": ft,
                    "funding_rate": float(row["fundingRate"]),
                    "mark_price":   float(row.get("markPrice", 0)),
                })
            df   = pd.DataFrame(rows)
            path = SUPP_DIR / f"funding_{pair}.csv"
            if path.exists() and len(df) > 0:
                old = pd.read_csv(path)
                df  = pd.concat([old, df]).drop_duplicates(
                    "funding_time").sort_values("funding_time")
                # Keep last 30 days
                cutoff = _ts_now_ms() - 750 * 3600 * 1000
                df = df[df["funding_time"] >= cutoff]
            if len(df) > 0:
                df.to_csv(path, index=False)
            result[token] = df
            time.sleep(0.1)
        except Exception as e:
            print(f"  Funding {token}: {e}")
    print(f"  Funding rates: {len(result)} tokens updated")
    return result


# ══════════════════════════════════════════════════════════════════════════════
# DOMINANCE + MARKET CAP — CoinGecko
# ══════════════════════════════════════════════════════════════════════════════

def fetch_dominance_mcap() -> pd.DataFrame:
    """Fetch BTC dominance + market caps from CoinGecko (last 30 days)."""
    CG_IDS = {
        "BTC": "bitcoin", "BNB": "binancecoin", "SOL": "solana",
        "ETH": "ethereum", "XRP": "ripple", "INJ": "injective-protocol",
        "DOGE": "dogecoin", "LTC": "litecoin",
    }
    rows_by_date = {}
    for token, cg_id in CG_IDS.items():
        try:
            url = (f"https://api.coingecko.com/api/v3/coins/{cg_id}/market_chart"
                   f"?vs_currency=usd&days=30&interval=daily")
            r = requests.get(url, timeout=30)
            for ts_ms, mcap in r.json().get("market_caps", []):
                key = datetime.datetime.fromtimestamp(
                    ts_ms/1000, tz=UTC).strftime("%Y-%m-%d")
                rows_by_date.setdefault(key, {"date": key})[f"mcap_{token}"] = mcap
            time.sleep(1.2)
        except Exception as e:
            print(f"  CoinGecko {token}: {e}")

    sorted_rows = sorted(rows_by_date.values(), key=lambda x: x["date"])
    for row in sorted_rows:
        total = sum(row.get(f"mcap_{t}", 0) for t in ALL_TOKENS)
        btc   = row.get("mcap_BTC", 0)
        row["total_mcap_8tokens"] = total
        row["btc_dominance_pct"]  = round(btc / total * 100, 4) if total else 0

    df   = pd.DataFrame(sorted_rows)
    path = SUPP_DIR / "dominance_marketcap.csv"
    if path.exists() and len(df) > 0:
        old = pd.read_csv(path)
        df  = pd.concat([old, df]).drop_duplicates("date").sort_values("date")
    if len(df) > 0:
        df.to_csv(path, index=False)
    print(f"  Dominance/MCap: {len(df)} rows")
    return df


# ══════════════════════════════════════════════════════════════════════════════
# ON-CHAIN PROXY — derived from OHLCV (same as training)
# ══════════════════════════════════════════════════════════════════════════════

SUPPLY = {
    "BTC": 19_700_000, "BNB": 145_000_000, "SOL": 470_000_000,
    "ETH": 120_000_000, "XRP": 57_000_000_000, "INJ": 100_000_000,
    "DOGE": 145_000_000_000, "LTC": 74_000_000,
}

def update_onchain_from_ohlcv():
    """Regenerate on-chain proxy CSVs from latest OHLCV data."""
    for token in ALL_TOKENS:
        raw_path = RAW_DIR / f"{token}USDT_1h_live.csv"
        out_path = SUPP_DIR / f"onchain_{token}.csv"
        if not raw_path.exists():
            continue
        try:
            df = pd.read_csv(raw_path, index_col=0, parse_dates=True)
            df.index = pd.to_datetime(df.index, utc=True)
            df["date"] = df.index.date.astype(str)
            daily = df.groupby("date").agg(
                close=("close", "last"),
                volume_usdt=("quote_volume", "sum"),
                trades=("trades", "sum"),
            ).reset_index()
            supply = SUPPLY.get(token, 1e9)
            daily["approx_mcap"]    = daily["close"].astype(float) * supply
            daily["vol_mcap_ratio"] = (
                daily["volume_usdt"].astype(float) /
                daily["approx_mcap"].replace(0, np.nan)).fillna(0)
            daily["pct_change_1d"]  = (
                daily["close"].astype(float).pct_change().fillna(0) * 100)
            daily = daily[["date","close","volume_usdt","approx_mcap",
                           "vol_mcap_ratio","pct_change_1d","trades"]]
            daily.to_csv(out_path, index=False)
        except Exception as e:
            print(f"  On-chain {token}: {e}")
    print(f"  On-chain proxy: updated for {len(ALL_TOKENS)} tokens")


# ══════════════════════════════════════════════════════════════════════════════
# SOCIAL INTEREST — CoinGecko Volume/MCap Ratio (replaces Google Trends)
# Volume/MCap ratio is a reliable proxy for social interest and search demand.
# High ratio = token is getting attention. Free, no API key, no rate limits.
# Saved as google_trends.csv for seamless Phase 1 compatibility.
# ══════════════════════════════════════════════════════════════════════════════

CG_IDS = {
    "BTC":  "bitcoin",        "BNB":  "binancecoin",
    "SOL":  "solana",         "ETH":  "ethereum",
    "XRP":  "ripple",         "INJ":  "injective-protocol",
    "DOGE": "dogecoin",       "LTC":  "litecoin",
}

def fetch_social_interest():
    """
    Fetch social interest proxy from CoinGecko (free, no key needed).
    Uses volume/mcap ratio normalized to 0-100 scale.
    Saved as google_trends.csv for Phase 1 feature engineering compatibility.
    Updates every 4h cycle.
    """
    try:
        ids_str = ",".join(CG_IDS.values())
        r = requests.get(
            "https://api.coingecko.com/api/v3/coins/markets",
            params={
                "vs_currency": "usd",
                "ids":         ids_str,
                "order":       "market_cap_desc",
                "per_page":    10,
                "sparkline":   "false",
                "price_change_percentage": "7d",
            },
            timeout=15,
        )
        coins = r.json()
        id_to_sym = {v: k for k, v in CG_IDS.items()}

        interest = {}
        for c in coins:
            sym       = id_to_sym.get(c["id"], c["symbol"].upper())
            vol       = float(c.get("total_volume", 0))
            mc        = float(c.get("market_cap", 1)) or 1
            # vol/mcap > 10% = very high interest (100), < 0.5% = low (0)
            vol_ratio = vol / mc * 100
            score     = min(100, max(0, round(vol_ratio * 10, 1)))
            interest[sym] = score

        # Build weekly-format DataFrame (13 rows) — same schema as old google_trends.csv
        today = datetime.datetime.now(UTC).strftime("%Y-%m-%d")
        dates = pd.date_range(
            end=today, periods=13, freq="W").strftime("%Y-%m-%d").tolist()
        df = pd.DataFrame(
            {sym: [interest.get(sym, 50)] * 13 for sym in ALL_TOKENS},
            index=dates,
        )
        df.index.name = "date"

        path = SUPP_DIR / "google_trends.csv"
        df.to_csv(path)
        print(f"  Social interest (CoinGecko vol/mcap): " +
              " | ".join([f"{s}={interest.get(s,0):.0f}" for s in ALL_TOKENS]))
        return df

    except Exception as e:
        print(f"  Social interest fetch failed: {e}")
        return pd.DataFrame()


# ══════════════════════════════════════════════════════════════════════════════
# MAIN FETCH CYCLE — called every 1h
# ══════════════════════════════════════════════════════════════════════════════

def run_fetch_cycle(full: bool = False):
    """
    Run one data fetch cycle.
    full=True: fetch everything including slow sources (run once at startup)
    full=False: fast cycle — OHLCV + funding only (runs every hour)
    """
    print(f"\n[{datetime.datetime.now(UTC).strftime('%Y-%m-%d %H:%M')} UTC] "
          f"Fetching live data (full={full})")

    fetch_all_ohlcv(hours=1000)      # always fetch OHLCV
    fetch_funding_rates(lookback_hours=1000)
    update_onchain_from_ohlcv()      # derived, always update

    if full:
        fetch_fear_greed()
        fetch_dominance_mcap()
        fetch_social_interest()   # CoinGecko vol/mcap ratio — updates every cycle

    print("  Data fetch complete")


def load_latest_ohlcv() -> Dict[str, pd.DataFrame]:
    """
    Load the latest saved OHLCV data for all tokens.
    Returns dict: token → DataFrame with last 750h of 1h candles.
    """
    data = {}
    for token in ALL_TOKENS:
        path = RAW_DIR / f"{token}USDT_1h_live.csv"
        if path.exists():
            df = pd.read_csv(path, index_col=0, parse_dates=True)
            df.index = pd.to_datetime(df.index, utc=True)
            data[token] = df
        else:
            # Fallback: fetch now
            df = fetch_ohlcv_binance(token, hours=1000)
            if not df.empty:
                df.to_csv(path)
                data[token] = df
    return data


if __name__ == "__main__":
    print("Running full data fetch...")
    run_fetch_cycle(full=True)
    print("Done.")
