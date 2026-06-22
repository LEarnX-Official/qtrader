"""
qtrader — HUD Monitor (sci-fi / cyberpunk edition)
A futuristic single-screen dashboard. Quit with Ctrl+C.

Reuses all data logic from monitor.py — this is purely a different SKIN.
Designed to fit an 80x24 terminal; adapts to larger screens automatically.

Usage:
  ./run.sh monitor_hud.py                # refresh every 5s
  ./run.sh monitor_hud.py --interval 3
  ./run.sh monitor_hud.py --no-boot      # skip the boot-up animation
  ./run.sh monitor_hud.py --live         # force on-chain baseline (live mode)
"""

import sys, os, time, datetime, argparse, itertools
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE_DIR))

UTC = datetime.timezone.utc

try:
    from rich.console import Console, Group
    from rich.table import Table
    from rich.panel import Panel
    from rich.layout import Layout
    from rich.live import Live
    from rich.text import Text
    from rich.align import Align
    from rich import box
except ImportError:
    os.system(f"{sys.executable} -m pip install rich -q")
    from rich.console import Console, Group
    from rich.table import Table
    from rich.panel import Panel
    from rich.layout import Layout
    from rich.live import Live
    from rich.text import Text
    from rich.align import Align
    from rich import box

# Reuse the proven data layer from the standard monitor.
import monitor as M

console = Console()

ELIGIBLE = M.ELIGIBLE_TOKENS
INELIG   = M.INELIGIBLE

# ── Sci-fi helpers ──────────────────────────────────────────────────────────

NEON       = ["#00e5ff", "#22d3ee", "#67e8f9", "#a78bfa", "#e879f9"]  # cyan→magenta
SPARK      = "▁▂▃▄▅▆▇█"
SPINNER    = itertools.cycle("⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏")
SCAN_FRAMES = ["▰▰▱▱▱▱", "▱▰▰▱▱▱", "▱▱▰▰▱▱", "▱▱▱▰▰▱",
               "▱▱▱▱▰▰", "▰▱▱▱▱▰"]
_frame = itertools.count()

# rolling price history for sparklines: token -> [recent prices]
_hist = {t: [] for t in ELIGIBLE}


def gradient(text: str, colors=NEON) -> Text:
    """Color a string char-by-char across a neon gradient."""
    t = Text()
    n = max(len(text) - 1, 1)
    for i, ch in enumerate(text):
        c = colors[int(i / n * (len(colors) - 1))]
        t.append(ch, style=f"bold {c}")
    return t


def sparkline(vals, width=10) -> str:
    if not vals:
        return "[dim]" + "·" * width + "[/]"
    v = vals[-width:]
    mn, mx = min(v), max(v)
    rng = mx - mn or 1e-9
    out = ""
    for x in v:
        idx = int((x - mn) / rng * (len(SPARK) - 1))
        out += SPARK[idx]
    out = out.rjust(width, "▁")
    color = "green" if v[-1] >= v[0] else "red"
    return f"[{color}]{out}[/]"


def led(on: bool, color="green") -> str:
    """A blinking status LED — pulses bright/dim by frame."""
    pulse = (next(itertools.islice(_frame_seq, 0, 1)) % 2 == 0)
    if not on:
        return "[red]●[/]"
    return f"[bold {color}]◉[/]" if pulse else f"[{color}]●[/]"


# a separate frame counter that doesn't get consumed by spinner
_frame_seq = itertools.count()


def bar(pct, width=10, color="cyan") -> str:
    fill = int(min(max(pct, 0), 1) * width)
    return f"[{color}]" + "█" * fill + "[/]" + "[dim]" + "░" * (width - fill) + "[/]"


# ── Boot sequence ───────────────────────────────────────────────────────────

def boot_sequence():
    lines = [
        ("loading neural core",        "OK"),
        ("connecting BSC mainnet",     "OK"),
        ("TWAK self-custody link",     "ARMED"),
        ("x402 payment channel",       "LIVE"),
        ("risk guardrails",            "ENGAGED"),
        ("market data feed",           "STREAMING"),
    ]
    console.clear()
    console.print()
    console.print(Align.center(gradient("◢◤  Q T R A D E R   O S   v1.0  ◢◤")))
    console.print(Align.center("[dim]autonomous self-custodial trading core[/]"))
    console.print()
    for label, status in lines:
        dots = "." * (28 - len(label))
        for _ in range(3):
            sp = next(SPINNER)
            console.print(f"  [cyan]{sp}[/] {label} [dim]{dots}[/]",
                          end="\r", soft_wrap=True)
            time.sleep(0.05)
        sc = {"OK": "green", "ARMED": "yellow", "LIVE": "cyan",
              "ENGAGED": "green", "STREAMING": "magenta"}.get(status, "green")
        console.print(f"  [green]✔[/] {label} [dim]{dots}[/] [bold {sc}]{status}[/]")
        time.sleep(0.08)
    console.print()
    console.print(Align.center("[bold green]◢◤◢◤  SYSTEM ONLINE  ◢◤◢◤[/]"))
    time.sleep(0.6)


# ── Panels ──────────────────────────────────────────────────────────────────

def hud_header(stats):
    f = next(_frame)
    ret = stats["total_return"]
    rc = "green" if ret >= 0 else "red"
    arrow = "▲" if ret >= 0 else "▼"
    now = datetime.datetime.now(UTC).strftime("%H:%M:%S")

    # countdown to trading window
    dnow = datetime.datetime.now(UTC)
    ts = datetime.datetime(2026, 6, 22, tzinfo=UTC)
    te = datetime.datetime(2026, 6, 28, 23, 59, tzinfo=UTC)
    if dnow < ts:   lbl, diff = "T-START", ts - dnow
    elif dnow < te: lbl, diff = "T-END",   te - dnow
    else:           lbl, diff = "ENDED",   None
    cd = (f"{diff.days}d {diff.seconds//3600:02d}h {(diff.seconds%3600)//60:02d}m"
          if diff else "—")

    try:
        from config import DRY_RUN
        mode = "[yellow]◇ PAPER[/]" if DRY_RUN else "[bold red]◆ LIVE[/]"
    except Exception:
        mode = "[yellow]◇ PAPER[/]"

    blink = "◉" if f % 2 == 0 else "●"
    title = gradient(" Q T R A D E R ")
    line2 = Text.from_markup(
        f"  [dim]CAP[/] [bold white]${stats['capital']:.2f}[/]   "
        f"[dim]PNL[/] [bold {rc}]{ret:+.2%} {arrow}[/]   "
        f"{mode}   "
        f"[dim]{lbl}[/] [bold magenta]{cd}[/]   "
        f"[dim]{now}[/]  [green]{blink}[/]")

    head = Text.assemble("◢◤", title, "◢◤  ",
                         (f"AI TRADING CORE", "bold cyan"),
                         ("   ◢◤◢◤", "magenta"))
    return Panel(Group(Align.center(head), line2),
                 box=box.DOUBLE_EDGE, border_style="bright_cyan",
                 padding=(0, 1))


def hud_assets(prices, stats):
    cap = stats["capital"]
    t = Table(box=box.SIMPLE_HEAD, show_header=True, padding=(0, 1),
              expand=True, border_style="dim cyan")
    t.add_column("◈ ASSET", style="bold", width=7)
    t.add_column("PRICE", justify="right", width=11)
    t.add_column("24H", justify="right", width=8)
    t.add_column("PULSE", justify="left", width=11)
    t.add_column("ALLOC", justify="left", width=12)
    t.add_column("VALUE", justify="right", width=9)

    for tok in ELIGIBLE:
        p = prices.get(tok, {})
        price, chg = p.get("price", 0), p.get("change", 0)
        if price:
            _hist[tok].append(price)
            _hist[tok][:] = _hist[tok][-30:]
        w = stats["weights"].get(tok, 0.0)
        val = cap * w
        cc = "green" if chg >= 0 else "red"
        wc = "cyan" if w > 0.01 else "dim"
        t.add_row(
            f"[{wc}]▸ {tok}[/]",
            f"${price:,.4f}" if price else "[dim]—[/]",
            f"[{cc}]{chg:+.1f}%[/]",
            sparkline(_hist[tok], 9),
            bar(w, 8, wc if w > 0.01 else "dim") + f" [{wc}]{w:.0%}[/]",
            f"[{wc}]${val:.2f}[/]" if w > 0.001 else "[dim]$0[/]",
        )

    cashv = cap * stats["cash_weight"]
    t.add_row("[yellow]▸ USDC[/]", "[dim]$1.00[/]", "[dim]—[/]",
              "[dim]▔▔▔▔▔▔▔▔▔[/]",
              bar(stats["cash_weight"], 8, "yellow") + f" [yellow]{stats['cash_weight']:.0%}[/]",
              f"[yellow]${cashv:.2f}[/]")

    return Panel(t, title="[bold cyan]▣ MARKET MATRIX[/]",
                 box=box.HEAVY, border_style="cyan", padding=(0, 1))


def hud_metrics(stats):
    ret, mdd, sh = stats["total_return"], stats["max_drawdown"], stats["sharpe"]
    rc = "green" if ret >= 0 else "red"
    sc = "bold green" if sh > 3 else "green" if sh > 1 else "yellow" if sh > 0 else "red"
    t = Table(box=None, show_header=False, padding=(0, 0), expand=True)
    t.add_column(style="dim", no_wrap=True); t.add_column(justify="right", no_wrap=True)
    t.add_row("CAP", f"[bold white]${stats['capital']:.2f}[/]")
    t.add_row("PNL", f"[bold {rc}]{ret:+.2%}[/]")
    t.add_row("SHRP", f"[{sc}]{sh:.2f}[/]")
    t.add_row("TRDS", f"[cyan]{stats['n_trades']}[/]")
    t.add_row("x402", f"[dim]${stats['x402_spent']:.3f}[/]")
    return Panel(t, title="[bold green]▣ STATS[/]",
                 box=box.HEAVY, border_style="green", padding=(0, 1))


def hud_risk(stats):
    mdd = stats["max_drawdown"]; L = 16
    wp, sp = int(0.10/0.30*L), int(0.15/0.30*L)
    fill = int(min(mdd/0.30, 1.0)*L)
    b = ""
    for i in range(L):
        if i < fill:
            b += "[bold red]█[/]" if i >= sp else "[yellow]█[/]" if i >= wp else "[green]█[/]"
        elif i == wp: b += "[dim yellow]┊[/]"
        elif i == sp: b += "[dim red]┊[/]"
        else: b += "[dim]░[/]"
    st = ("[bold red]⚠ STOP[/]" if mdd >= 0.15 else
          "[bold yellow]⚠ WARN[/]" if mdd >= 0.10 else
          "[bold green]✓ SECURE[/]")
    bs = "red" if mdd >= 0.15 else "yellow" if mdd >= 0.10 else "green"
    scan = SCAN_FRAMES[next(_frame_seq) % len(SCAN_FRAMES)]
    txt = (f"  {b}\n"
           f"  DD [bold]{mdd:.1%}[/]  {st}\n"
           f"  [dim]10▸15▸30 DQ[/]\n"
           f"  [cyan]SCAN[/] [magenta]{scan}[/]")
    return Panel(txt, title="[bold]▣ RISK CORE[/]", box=box.HEAVY,
                 border_style=bs, padding=(0, 1))


def hud_equity(df):
    if df.empty or len(df) < 2:
        return Panel("[dim]awaiting telemetry…[/]", title="[bold yellow]▣ EQUITY[/]",
                     box=box.HEAVY, border_style="yellow", padding=(0, 1))
    base = float(df["capital"].iloc[0]) or M.INITIAL_CAP
    caps = [base] + df["capital"].tolist()
    W, H = 46, 5
    mn, mx = min(caps), max(caps); rng = max(mx - mn, 0.01)
    step = max(1, len(caps)//W); s = caps[::step][-W:]
    rows = []
    for ri in range(H):
        row = ""
        for v in s:
            y = int((v-mn)/rng*(H-1)); cr = H-1-y
            by = int((base-mn)/rng*(H-1)); br = H-1-by
            if cr == ri: row += "[green]▰[/]" if v >= base else "[red]▰[/]"
            elif ri == br: row += "[dim cyan]┄[/]"
            else: row += " "
        rows.append(row)
    tr = (caps[-1]-base)/base; rc = "green" if tr >= 0 else "red"
    body = "\n".join(rows)
    return Panel(body, title=f"[bold yellow]▣ EQUITY[/] [{rc}]{tr:+.2%}[/]",
                 box=box.HEAVY, border_style="yellow", padding=(0, 1))


def hud_footer(interval):
    sp = next(SPINNER)
    f = next(_frame_seq)
    chain = "[green]◉[/]" if f % 2 == 0 else "[green]●[/]"
    txt = (f" [cyan]{sp}[/] [dim]live feed[/]  {chain}[dim]BSC[/]  "
           f"[green]●[/][dim]TWAK[/]  [green]●[/][dim]MODEL[/]  "
           f"[dim]│ refresh {interval}s │ Ctrl+C to disengage[/]")
    return Panel(txt, box=box.MINIMAL, border_style="dim cyan", padding=(0, 0))


# ── Layout (fits 80x24, grows on bigger screens) ────────────────────────────

def render(interval):
    df     = M.load_trades()
    prices = M.fetch_prices()
    stats  = M.get_stats(df)

    lay = Layout()
    lay.split_column(
        Layout(hud_header(stats),       name="head", size=4),
        Layout(hud_assets(prices, stats), name="assets", size=10),
        Layout(name="mid", size=7),
        Layout(hud_footer(interval),    name="foot", size=3),
    )
    lay["mid"].split_row(
        Layout(hud_metrics(stats), name="metrics", ratio=2),
        Layout(hud_risk(stats),    name="risk",    ratio=2),
        Layout(hud_equity(df),     name="equity",  ratio=3),
    )
    return lay


def run(interval, do_boot=True):
    if not sys.stdout.isatty():
        # non-TTY fallback: plain status line (so logs/pipes still work)
        while True:
            df = M.load_trades(); st = M.get_stats(df)
            print(f"[{datetime.datetime.now(UTC):%H:%M:%S}] "
                  f"CAP ${st['capital']:.2f} | PNL {st['total_return']:+.2%} | "
                  f"DD {st['max_drawdown']:.2%} | trades {st['n_trades']}",
                  flush=True)
            time.sleep(interval)

    if do_boot:
        boot_sequence()
    try:
        with Live(render(interval), console=console, refresh_per_second=4,
                  screen=True) as live:
            while True:
                # animate faster than data refresh for a "live" feel
                for _ in range(max(1, interval * 2)):
                    live.update(render(interval))
                    time.sleep(0.5)
    except KeyboardInterrupt:
        pass
    console.clear()
    console.print(gradient("◢◤  QTRADER CORE DISENGAGED  ◢◤"))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--interval", type=int, default=5)
    p.add_argument("--no-boot", action="store_true", help="skip boot animation")
    p.add_argument("--live", action="store_true",
                   help="force on-chain balance baseline (live mode)")
    a = p.parse_args()

    if a.live:
        import config
        config.DRY_RUN = False
        M.INITIAL_CAP = M._detect_baseline_capital()

    run(a.interval, do_boot=not a.no_boot)
