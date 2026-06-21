"""
Quantum Trader — Live Monitor
Stable auto-refresh dashboard. Quit with Ctrl+C.

Usage:
  ./run.sh monitor.py                # refresh every 15s
  ./run.sh monitor.py --interval 5   # refresh every 5s
"""

import sys, os, time, datetime, argparse
import numpy as np
import pandas as pd
import requests
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE_DIR))

UTC = datetime.timezone.utc

try:
    from rich.console import Console
    from rich.table import Table
    from rich.panel import Panel
    from rich.layout import Layout
    from rich.live import Live
    from rich import box
except ImportError:
    os.system(f"{sys.executable} -m pip install rich -q")
    from rich.console import Console
    from rich.table import Table
    from rich.panel import Panel
    from rich.layout import Layout
    from rich.live import Live
    from rich import box

console = Console()

TRADE_TOKENS    = ["BNB", "SOL", "ETH", "XRP", "INJ", "DOGE", "LTC"]
ELIGIBLE_TOKENS = ["ETH", "XRP", "INJ", "DOGE", "LTC"]   # competition-eligible
INELIGIBLE      = ["BNB", "SOL"]                          # not on official list
RESULTS_DIR     = BASE_DIR / "results"
INITIAL_CAP     = 100.0

# ── Data ──────────────────────────────────────────────────────────────────────

def load_trades() -> pd.DataFrame:
    path = RESULTS_DIR / "trades_live.csv"
    if not path.exists():
        return pd.DataFrame()
    try:
        df = pd.read_csv(path)
        df["datetime"] = pd.to_datetime(df["datetime"], format="mixed", utc=True)
        return df.sort_values("datetime").reset_index(drop=True)
    except Exception:
        return pd.DataFrame()


_price_cache = {"ts": 0, "data": {}}

def fetch_prices() -> dict:
    # Cache prices 10s to avoid hammering Binance on fast refresh
    if time.time() - _price_cache["ts"] < 10 and _price_cache["data"]:
        return _price_cache["data"]
    prices = {}
    try:
        syms = '["' + '","'.join(f"{t}USDT" for t in TRADE_TOKENS) + '"]'
        r = requests.get("https://api.binance.com/api/v3/ticker/24hr",
                         params={"symbols": syms}, timeout=6)
        for item in r.json():
            sym = item["symbol"].replace("USDT", "")
            if sym in TRADE_TOKENS:
                prices[sym] = {
                    "price":  float(item["lastPrice"]),
                    "change": float(item["priceChangePercent"]),
                    "high":   float(item["highPrice"]),
                    "low":    float(item["lowPrice"]),
                }
    except Exception:
        prices = {t: {"price": 0, "change": 0, "high": 0, "low": 0} for t in TRADE_TOKENS}
    _price_cache["ts"]   = time.time()
    _price_cache["data"] = prices
    return prices


def get_stats(df: pd.DataFrame) -> dict:
    if df.empty:
        return {"capital": INITIAL_CAP, "total_return": 0.0, "max_drawdown": 0.0,
                "sharpe": 0.0, "n_trades": 0, "cash_weight": 1.0,
                "weights": {t: 0.0 for t in TRADE_TOKENS},
                "last_trade": None, "last_reason": "—", "x402_spent": 0.0}
    latest  = df.iloc[-1]
    capital = float(latest["capital"])
    weights = {t: float(latest.get(f"w_{t}", 0.0)) for t in TRADE_TOKENS}
    x402    = float(df["x402_paid"].sum()) if "x402_paid" in df.columns else 0.0

    caps = np.array([INITIAL_CAP] + df["capital"].tolist())
    rets = np.diff(caps) / caps[:-1]
    peaks = np.maximum.accumulate(caps)
    mdd  = float(((peaks - caps) / np.maximum(peaks, 1e-8)).max())
    sharpe = float(rets.mean() / rets.std() * np.sqrt(365 * 24)) if len(rets) > 1 and rets.std() > 1e-9 else 0.0

    return {"capital": capital, "total_return": (capital - INITIAL_CAP) / INITIAL_CAP,
            "max_drawdown": mdd, "sharpe": sharpe, "n_trades": len(df),
            "cash_weight": float(latest.get("cash_weight", 1.0)), "weights": weights,
            "last_trade": pd.to_datetime(latest["datetime"]),
            "last_reason": str(latest.get("reason", "—"))[:20], "x402_spent": x402}


# ── Panels ────────────────────────────────────────────────────────────────────

def header(stats):
    now = datetime.datetime.now(UTC).strftime("%Y-%m-%d %H:%M:%S UTC")
    ret = stats["total_return"]; rc = "green" if ret >= 0 else "red"
    dnow = datetime.datetime.now(UTC)
    tstart = datetime.datetime(2026, 6, 22, tzinfo=UTC)
    tend   = datetime.datetime(2026, 6, 28, 23, 59, tzinfo=UTC)
    if dnow < tstart:   lbl, diff = "Trading starts in", tstart - dnow
    elif dnow < tend:   lbl, diff = "Trading ends in",   tend - dnow
    else:               lbl, diff = "Ended", None
    cd = f"{diff.days}d {diff.seconds//3600}h {(diff.seconds%3600)//60}m" if diff else "—"
    txt = (f"[bold cyan]  QUANTUM TRADER[/]  [dim]|[/]  [bold]BNB Hack AI Agent ⚡[/]  "
           f"[dim]|[/]  [dim]{now}[/]\n\n"
           f"  [dim]Capital:[/] [bold white]${stats['capital']:.2f}[/]   "
           f"[dim]Return:[/] [bold {rc}]{ret:+.2%}[/]   "
           f"[dim]Mode:[/] [yellow]📋 PAPER[/]   "
           f"[dim]{lbl}:[/] [bold magenta]{cd}[/]")
    return Panel(txt, border_style="blue", padding=(0,1))


def prices_panel(prices, stats):
    t = Table(box=box.SIMPLE_HEAD, show_header=True, padding=(0,2), expand=True)
    t.add_column("Token", style="bold", width=8)
    t.add_column("Price", justify="right")
    t.add_column("24h %", justify="right")
    t.add_column("High", justify="right")
    t.add_column("Low", justify="right")
    t.add_column("Weight", justify="right")
    t.add_column("Value", justify="right")
    cap = stats["capital"]

    # Eligible tokens first (tradeable)
    for tok in ELIGIBLE_TOKENS:
        p = prices.get(tok, {})
        price, chg = p.get("price", 0), p.get("change", 0)
        w = stats["weights"].get(tok, 0.0); val = cap * w
        cc = "green" if chg >= 0 else "red"
        wc = "cyan" if w > 0.01 else "dim"
        t.add_row(f"[{wc}]{tok}[/]",
                  f"${price:,.4f}" if price else "[dim]—[/]",
                  f"[{cc}]{chg:+.2f}%[/]",
                  f"[dim]${p.get('high',0):,.4f}[/]",
                  f"[dim]${p.get('low',0):,.4f}[/]",
                  f"[{wc}]{w:.1%}[/]" if w > 0.001 else "[dim]0%[/]",
                  f"[{wc}]${val:.2f}[/]" if w > 0.001 else "[dim]$0[/]")

    # Cash
    cashv = cap * stats["cash_weight"]
    t.add_row("[yellow]USDT[/]", "[dim]$1.00[/]", "[dim]—[/]", "[dim]—[/]", "[dim]—[/]",
              f"[yellow]{stats['cash_weight']:.1%}[/]", f"[yellow]${cashv:.2f}[/]")

    # Ineligible tokens shown dimmed (context only — never held)
    for tok in INELIGIBLE:
        p = prices.get(tok, {})
        price, chg = p.get("price", 0), p.get("change", 0)
        cc = "green" if chg >= 0 else "red"
        t.add_row(f"[dim strike]{tok}[/]",
                  f"[dim]${price:,.4f}[/]" if price else "[dim]—[/]",
                  f"[dim]{chg:+.2f}%[/]",
                  "[dim]—[/]", "[dim]—[/]",
                  "[red dim]✗ N/A[/]", "[dim]ineligible[/]")

    return Panel(t, title="[bold]Live Prices & Portfolio[/]  [dim](5 eligible + cash)[/]",
                 border_style="cyan", padding=(0,1))


def metrics_panel(stats):
    t = Table(box=box.SIMPLE, show_header=False, padding=(0,1), expand=True)
    t.add_column("", style="dim"); t.add_column("")
    t.add_column("", style="dim"); t.add_column("")
    ret, mdd, sh = stats["total_return"], stats["max_drawdown"], stats["sharpe"]
    rc = "green" if ret >= 0 else "red"
    dc = "green" if mdd < 0.10 else "yellow" if mdd < 0.15 else "red"
    sc = "bold green" if sh > 3 else "green" if sh > 1 else "yellow" if sh > 0 else "red"
    last = stats["last_trade"].strftime("%H:%M") if stats["last_trade"] else "—"
    t.add_row("💰 Capital", f"[bold white]${stats['capital']:.2f}[/]",
              "📈 Return", f"[bold {rc}]{ret:+.2%}[/]")
    t.add_row("📉 Max DD", f"[{dc}]{mdd:.2%}[/]", "⚡ Sharpe", f"[{sc}]{sh:.2f}[/]")
    t.add_row("🔄 Trades", f"[cyan]{stats['n_trades']}[/]", "🕐 Last", f"[dim]{last}[/]")
    t.add_row("💵 Cash", f"[yellow]{stats['cash_weight']:.1%}[/]",
              "💳 x402", f"[dim]${stats['x402_spent']:.4f}[/]")
    t.add_row("🎯 TP", "[dim]+10% / +20%[/]", "🛑 Trail", "[dim]-5% peak[/]")
    return Panel(t, title="[bold]Metrics[/]", border_style="green", padding=(0,1))


def risk_panel(stats):
    mdd = stats["max_drawdown"]; L = 24
    wp, sp = int(0.10/0.30*L), int(0.15/0.30*L)
    fill = int(min(mdd/0.30, 1.0)*L)
    bar = ""
    for i in range(L):
        if i < fill:
            bar += "[bold red]█[/]" if i >= sp else "[yellow]█[/]" if i >= wp else "[green]█[/]"
        elif i == wp: bar += "[dim yellow]│[/]"
        elif i == sp: bar += "[dim red]│[/]"
        else: bar += "[dim]░[/]"
    st = "[bold red]🚨 STOP[/]" if mdd >= 0.15 else "[bold yellow]⚠️ WARN[/]" if mdd >= 0.10 else "[bold green]✅ OK[/]"
    txt = f"  {bar}\n  DD: [bold]{mdd:.2%}[/]  {st}\n  [dim]10%warn 15%stop 30%DQ[/]"
    bs = "red" if mdd >= 0.15 else "yellow" if mdd >= 0.10 else "green"
    return Panel(txt, title="[bold]Risk[/]", border_style=bs, padding=(0,1))


def equity_panel(df):
    if df.empty or len(df) < 2:
        return Panel("[dim]Equity curve after first trades...[/]",
                     title="[bold]Equity Curve[/]", border_style="yellow")
    caps = [INITIAL_CAP] + df["capital"].tolist()
    W, H = 50, 6
    mn, mx = min(caps), max(caps); rng = max(mx-mn, 0.01)
    step = max(1, len(caps)//W); s = caps[::step][-W:]
    rows = []
    for ri in range(H):
        row = ""
        for v in s:
            y = int((v-mn)/rng*(H-1)); cr = H-1-y
            by = int((INITIAL_CAP-mn)/rng*(H-1)); br = H-1-by
            if cr == ri: row += "[green]█[/]" if v >= INITIAL_CAP else "[red]█[/]"
            elif ri == br: row += "[dim]─[/]"
            else: row += " "
        rows.append(row)
    tr = (caps[-1]-INITIAL_CAP)/INITIAL_CAP; rc = "green" if tr >= 0 else "red"
    body = f"[dim]${mx:.2f}[/]\n" + "\n".join(rows) + f"\n[dim]${mn:.2f}[/]"
    return Panel(body, title=f"[bold]Equity[/] [{rc}]{tr:+.2%}[/] [dim]({len(df)}t)[/]",
                 border_style="yellow", padding=(0,1))


def trades_panel(df, n=8):
    if df.empty:
        return Panel("[dim]No trades yet...[/]", title="[bold]Recent Trades[/]",
                     border_style="magenta")
    t = Table(box=box.SIMPLE_HEAD, show_header=True, padding=(0,1), expand=True)
    t.add_column("Time",   width=14)
    t.add_column("Capital",justify="right", width=10)
    t.add_column("Return", justify="right", width=10)
    t.add_column("Cash",   justify="right", width=7)
    # All 5 eligible token weight columns
    for tok in ELIGIBLE_TOKENS:
        t.add_column(tok, justify="right", width=6)
    t.add_column("Trades", justify="right", width=7)
    t.add_column("Reason", width=22)

    for _, r in df.tail(n).iloc[::-1].iterrows():
        try:
            dt  = pd.to_datetime(r["datetime"]).strftime("%m/%d %H:%M:%S")
            cap = float(r["capital"]); ret = float(r.get("port_return", 0))
            csh = float(r.get("cash_weight", 1))
            ntr = int(r.get("n_trades", 0))
            rsn = str(r.get("reason", ""))[:22]
            rc  = "green" if ret >= 0 else ("red" if ret < 0 else "dim")

            row = [
                f"[dim]{dt}[/]",
                f"[white]${cap:.2f}[/]",
                f"[{rc}]{ret:+.4f}[/]",
                f"[yellow]{csh:.0%}[/]",
            ]
            for tok in ELIGIBLE_TOKENS:
                w = float(r.get(f"w_{tok}", 0))
                cell = f"[cyan]{w:.0%}[/]" if w > 0.01 else "[dim]·[/]"
                row.append(cell)
            row.append(f"[magenta]{ntr}[/]" if ntr > 0 else "[dim]0[/]")
            row.append(f"[dim]{rsn}[/]")
            t.add_row(*row)
        except Exception:
            continue
    return Panel(t, title="[bold]Recent Trades[/]  [dim](full detail)[/]",
                 border_style="magenta", padding=(0,1))


def footer(interval):
    now = datetime.datetime.now(UTC).strftime("%H:%M:%S UTC")
    return Panel(f"[dim]  Refresh {interval}s  │  Ctrl+C to quit  │  {now}[/]",
                 border_style="dim", padding=(0,0))


def render(interval):
    df     = load_trades()
    prices = fetch_prices()
    stats  = get_stats(df)
    lay = Layout()
    lay.split_column(
        Layout(header(stats), name="h", size=4),
        Layout(prices_panel(prices, stats), name="p", size=11),
        Layout(name="m", size=8),
        Layout(equity_panel(df), name="e", size=9),
        Layout(trades_panel(df, n=6), name="t"),   # flexible — absorbs leftover space
        Layout(footer(interval), name="f", size=3),
    )
    lay["m"].split_row(Layout(metrics_panel(stats), ratio=3),
                       Layout(risk_panel(stats), ratio=2))
    return lay


def run_monitor(interval=15):
    try:
        with Live(render(interval), console=console, refresh_per_second=1,
                  screen=True) as live:
            while True:
                time.sleep(interval)
                live.update(render(interval))
    except KeyboardInterrupt:
        pass
    console.clear()
    console.print("[bold green]Monitor stopped.[/]")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--interval", type=int, default=15)
    a = p.parse_args()
    run_monitor(a.interval)
