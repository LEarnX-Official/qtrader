# qtrader — Autonomous AI Trading Agent ⚡

> **BNB Hack: AI Trading Agent Edition** — BNB Chain × CoinMarketCap × Trust Wallet
> **Track 1 — Autonomous Trading Agents** · $24,000 (5 winners) of a $36,000 pool
> Live trading window: **June 22–28, 2026** · Scored on real on-chain PnL

`qtrader` is a **crypto-native, self-custodial AI trading agent** for BNB Chain
(BSC). It reads live markets, **pays for premium data per request via real x402
micropayments (USDC on Base)**, decides with a proprietary ML brain, enforces
hard risk guardrails, and **signs and broadcasts its own BEP-20 swaps** on BSC
through the **Trust Wallet Agent Kit (TWAK)** — keys never leave the device. It
is built to run genuinely hands-off for the full competition week without
blowing past the 30% drawdown disqualification gate.

**Dual-chain by design:** the agent **pays for data on Base** (where x402
settles in USDC) and **trades on BSC** (the competition chain) — both from the
same self-custodial wallet.

---

## What I'm Building Here

The hackathon's premise: *"natural-language strategy in, on-chain execution
out"* — an agent that reads markets via CMC, decides, and signs its own txs via
TWAK, then trades live on BSC and is ranked by **total return under a max-drawdown
risk gate**.

`qtrader` is my Track 1 entry, and it deliberately targets all three special
prizes at once:

- **The data layer** pays for premium market data **per request via real x402
  micropayments** — genuine USDC settlement on Base, signed by the agent wallet,
  returning live funding-rate / derivatives / sentiment data that feeds the
  decision loop. Not a README mention — verifiable on-chain. → *Best Use of
  Agent Hub + native x402.*
- **The execution layer** is TWAK and nothing else — local key signing,
  autonomous mode, x402, and on-chain guardrails run end-to-end. No custodial
  step anywhere in the trade loop. → *Best Use of TWAK.*
- **The identity layer** registers the strategy on-chain via the BNB AI
  Agent SDK and serves it as a tradeable agent endpoint. → *Best Use of BNB AI
  Agent SDK.*

The headline edge is the **brain**: rather than a thin LLM wrapper around a swap
call, the decision engine runs on trained model weights produced by our **main
architecture** — a separate, proprietary research stack we don't detail here.
`qtrader` is the live serving layer around those weights. What matters in
practice is the behavior they produce: the agent stays **defensive and
cash-heavy when conviction is low**, which is exactly what survives a
drawdown-capped competition.

> 📩 **Want to talk about the full research stack** (the model architecture,
> training methodology, and how the weights are produced)? That work is private,
> but I'm happy to discuss it directly — **contact me at
> ritwickhaldar2018@gmail.com**.

---

## Table of Contents

1. [Main Architecture](#main-architecture)
2. [Dual-Chain: Pay on Base, Trade on BSC](#dual-chain-pay-on-base-trade-on-bsc)
3. [Real x402 Micropayments](#real-x402-micropayments)
4. [The Decision Brain](#the-decision-brain)
5. [The Trading Cycle](#the-trading-cycle)
6. [Backtest Results](#backtest-results)
7. [Project Structure](#project-structure)
8. [Setup](#setup)
9. [Running](#running)
10. [Risk & Profit Management](#risk--profit-management)
11. [Token Universe](#token-universe)
12. [Configuration Reference](#configuration-reference)
13. [Monitoring & Alerts](#monitoring--alerts)
14. [Cron Deployment](#cron-deployment)
15. [Paper → Live Switch](#paper--live-switch)
16. [Tech Stack](#tech-stack)
17. [Special Prize Coverage](#special-prize-coverage)
18. [Competition Details](#competition-details)
19. [Contact](#contact)

---

## Main Architecture

`qtrader` is organized into four clean layers. Data flows top-to-bottom every
cycle; the orchestrator in [agent.py](agent.py) is the only thing that knows
about all four.

```
┌──────────────────────────────────────────────────────────────────────┐
│  ORCHESTRATOR — agent.py                                               │
│  7-step cycle · dual cadence (15min price-check / 1h full inference)   │
└───────────────┬──────────────────────────────────────────────────────┘
                │
   ┌────────────▼─────────────┐   1. DATA LAYER
   │ data/cmc_hub.py          │   • CMC Agent Hub (quotes, F&G, dominance)
   │ data/fetcher.py          │   • Binance public API (OHLCV fallback)
   │                          │   • x402 micropayment per request
   └────────────┬─────────────┘
                │ OHLCV (168h) + supplementary features
   ┌────────────▼─────────────┐   2. DECISION LAYER (the "brain")
   │ inference/engine.py      │   • Loads trained weights from the main
   │ decision/cmc_decision.py │     architecture → portfolio weights
   └────────────┬─────────────┘   → target portfolio weights + cash weight
                │ weights
   ┌────────────▼─────────────┐   3. RISK LAYER (guardrails)
   │ risk/manager.py          │   • Drawdown warn/stop, 40% position cap
   │ risk/profit_manager.py   │   • Per-token take-profit + trailing stop
   └────────────┬─────────────┘   → adjusted, safe weights
                │ approved weights
   ┌────────────▼─────────────┐   4. EXECUTION LAYER (self-custody)
   │ execution/trader.py      │   • Rebalance math, portfolio state, trade log
   │ execution/twak_client.py │   • TWAK local signing + on-chain guardrails
   │ execution/bnb_agent.py   │   • BNB Agent SDK on-chain identity + endpoint
   └────────────┬─────────────┘   → signed BEP-20 swaps on PancakeSwap V2 (BSC)
                │
   ┌────────────▼─────────────┐   OBSERVABILITY
   │ alerts/telegram.py       │   • trade / risk / daily-summary alerts
   │ monitor.py               │   • live terminal dashboard
   │ results/trades_live.csv  │   • every-cycle snapshot (holds included)
   └──────────────────────────┘
```

**Design principles baked into the architecture:**

- **Separation of concerns.** Data, decision, risk, and execution are
  independent packages. The brain produces *intent* (weights); risk *vetoes or
  shrinks* it; execution *signs and broadcasts* it. No layer reaches across.
- **Self-custody, end to end.** Signing happens only in
  [twak_client.py](execution/twak_client.py), locally, for every trade — there
  is no point in the loop where a third party holds keys or co-signs.
- **Fail safe, not open.** If inference, data, or a swap fails, the cycle logs
  the error, alerts, and **leaves portfolio state unchanged** — no phantom
  capital moves, no false trades recorded.
- **Models are external & reused.** The trained weights come from our **main
  architecture** (a separate research stack), loaded **once** at startup via
  `QT_DIR` in [config.py](config.py#L26); this repo is the live serving layer
  around them.
- **One switch from paper to live.** `DRY_RUN` flips logging into real on-chain
  execution with zero other changes.

---

## Dual-Chain: Pay on Base, Trade on BSC

qtrader deliberately spans **two chains**, using each for what it does best —
both driven by the **same self-custodial wallet** (`0x4547C17B…`):

```
        ┌──────────────────────── EACH CYCLE ────────────────────────┐
        │                                                            │
   ┌────▼─────────────────────────┐        ┌────────────────────────▼────┐
   │  PAY FOR DATA → x402 on BASE  │        │  TRADE → swaps on BSC        │
   │  USDC on Base mainnet (8453)  │        │  USDC ↔ tokens on chain 56   │
   │  real EIP-3009 authorization  │        │  PancakeSwap V2              │
   │  ← funding rates, derivatives,│        │  competition wallet          │
   │    sentiment, on-chain flow   │        │  TWAK local signing          │
   └───────────────────────────────┘        └──────────────────────────────┘
            (data fees)                              (capital at work)
```

| Rail | Chain | Asset | Why |
|------|-------|-------|-----|
| **Data fees (x402)** | Base mainnet (8453) | USDC | x402 settles in USDC on Base (Coinbase facilitator) |
| **Trading (swaps)** | BSC (56) | USDC base | The competition is scored on BSC PnL |

> Why split chains? x402's mature payment rail lives on Base; the competition
> lives on BSC. Rather than force one chain to do both, qtrader pays data fees
> where x402 actually settles (Base) and keeps all trading capital on BSC where
> it's scored. One wallet, two rails, no custodial step on either.

---

## Real x402 Micropayments

x402 is the open **HTTP-402 payment standard** (Coinbase) — a server replies
`402 Payment Required`, the client signs a stablecoin payment authorization, and
the request is retried with proof of payment. qtrader uses it for **real**, not
as a label.

**How it works in qtrader** ([execution/x402_payments.py](execution/x402_payments.py)):

1. The agent calls a live x402-gated data endpoint on Base.
2. The endpoint returns `402` with payment requirements (USDC on Base).
3. **TWAK signs a real EIP-3009 USDC authorization** with the agent wallet —
   self-custodial, capped by a per-call budget guard so an unattended agent can
   never overspend.
4. The request is retried with the `X-PAYMENT` header → the endpoint returns
   live market data (funding-rate arbitrage, derivatives positioning, sentiment)
   that feeds the decision loop.

**Verifiable, not simulated.** Each call moves real USDC on Base:

```
USDC balance:  10.062371  →  10.061371  →  10.060371   (−0.001 per paid call)
```

Endpoints are configured in [config.py](config.py) (`X402_ENDPOINTS`), default
~$0.001/call, with a hard `X402_MAX_PAYMENT` cap. In `DRY_RUN` the signing path
is exercised but nothing broadcasts.

> This directly answers the prize criterion: *"The agent uses x402 to pay per
> request for data… as part of its trade loop. Real, not a README mention."*

---

## The Decision Brain

The decision brain ([inference/engine.py](inference/engine.py)) runs on **trained
model weights produced by our main architecture** — a separate, proprietary
research stack. Those details aren't published here; `qtrader` simply loads the
weights at startup and queries them each cycle.

What the brain takes in and returns:

| | |
|---|---|
| **In** | 168h (1-week) window of market features per token + live supplementary data |
| **Out** | Target portfolio weights + a cash weight |

A post-inference **minimum-weight filter** zeroes any position below 5% and
rolls it into cash, concentrating conviction and avoiding dust trades that waste
gas ([agent.py:116](agent.py#L116)).

> A more defensive alternative path,
> [decision/cmc_decision.py](decision/cmc_decision.py), drives allocation purely
> from CMC Agent Hub skills — market-regime detection plus per-token
> perp/CVD/positioning analysis — allocating only to tokens with confirmed
> bullish flow and holding cash otherwise. This both deepens Agent Hub usage and
> protects against the drawdown cap when conviction is low.

---

## The Trading Cycle

In `--loop` mode the agent runs two nested cadences (see
[agent.py:435](agent.py#L435)):

- **Every 15 min** — fast price check via Binance's free ticker, then the Profit
  Manager evaluates take-profit and trailing stops. If a target is hit, it
  rebalances and alerts immediately.
- **Every 1 h** — a full 7-step cycle: CMC data → inference → risk → profit
  check → TWAK guardrails → execute → alert.

Every cycle writes a snapshot to `results/trades_live.csv` — even a **hold** is
recorded, so the PnL curve and monitor stay continuous and the competition's
hourly scoring always sees deployed capital.

---

## Backtest Results

Backtested Jan–May 2026 on **unseen** data:

| Metric | qtrader | Equal-Weight Benchmark |
|--------|---------------:|-----------------------:|
| Total return | **+70.08%** | — |
| Sharpe ratio | **2.85** | **-0.55** (lost money) |
| Max drawdown | **5.87%** | — (well under the 30% gate) |

---

## Project Structure

```
qtrader/
├── agent.py                 ← Main entry point & 7-step cycle orchestrator
├── config.py                ← All settings, keys, addresses, thresholds
├── monitor.py               ← Live auto-refresh terminal dashboard
├── run.sh                   ← Launcher that pins the correct venv
├── requirements.txt
├── .env                     ← API keys & wallet secrets (not committed)
│
├── data/
│   ├── cmc_hub.py           ← CMC Agent Hub layer + x402 micropayments
│   ├── fetcher.py           ← CMC/Binance OHLCV + supplementary fetcher
│   ├── raw/                 ← Live OHLCV CSVs (1h candles)
│   ├── supplementary/       ← Fear/greed, funding, on-chain features
│   ├── processed/           ← Inference cache
│   └── logs/
│
├── inference/
│   └── engine.py            ← Inference engine (loads trained weights once)
│
├── decision/
│   └── cmc_decision.py      ← CMC-skill-driven defensive allocation layer
│
├── execution/
│   ├── trader.py            ← Portfolio state, rebalance math, trade logging
│   ├── twak_client.py       ← Trust Wallet Agent Kit: local signing + guardrails
│   └── bnb_agent.py         ← BNB AI Agent SDK (ERC-8004/8183) registration & server
│
├── risk/
│   ├── manager.py           ← Drawdown stops, position caps, min-trade rules
│   └── profit_manager.py    ← Per-token take-profit + trailing stop
│
├── alerts/
│   └── telegram.py          ← Trade / risk / daily-summary notifications
│
└── results/
    └── trades_live.csv      ← Per-cycle trade & snapshot log
```

---

## Setup

### 1. Install dependencies

```bash
# Python dependencies
pip install -r requirements.txt

# Trust Wallet Agent Kit CLI (Node.js, not pip) — installed globally as `twak`.
# Get it from the TWAK portal: https://portal.trustwallet.com
#   e.g.  npm install -g @trustwallet/twak     (see portal for the exact package)
twak --version   # verify it's on your PATH
```

### 2. Configure `.env`

```ini
CMC_API_KEY=...                # CoinMarketCap Pro API key
TWAK_API_KEY=...               # Trust Wallet Agent Kit key
AGENT_WALLET_ADDRESS=0x4547C17BEF404a61767DB061df14Fad7581D9aB1
WALLET_PASSWORD=...            # local key-signing password
TELEGRAM_BOT_TOKEN=...         # optional — enables alerts
TELEGRAM_CHAT_ID=...           # optional — enables alerts
# Optional perp data source:
ASTER_API_KEY=...
ASTER_API_SECRET=...
ASTER_USER_ADDRESS=...
```

Required keys (validated at startup): `CMC_API_KEY`, `TWAK_API_KEY`,
`AGENT_WALLET_ADDRESS`, `WALLET_PASSWORD`.

### 3. Verify config

```bash
python config.py
# → "Config OK — wallet=0x4547C17B...  dry_run=True"
```

---

## Running

All commands accept `--dry-run true|false` to override the mode in
[config.py](config.py#L81).

```bash
python agent.py                    # single cycle (for cron)
python agent.py --loop             # continuous (15min price / 1h inference)
python agent.py --status           # full component status report
python agent.py --register         # register on-chain (do before Jun 22!)
python agent.py --serve            # start the BNB Agent strategy server
python agent.py --loop --dry-run false   # force live mode
```

> Use `./run.sh agent.py --loop` to launch with the project's pinned venv.

### Registering for the competition

Track 1 registration is **on-chain** and closes when the trading window opens
(June 22). `python agent.py --register` resolves your agent wallet and submits
the registration tx via TWAK (`twak compete register`) and the BNB Agent SDK.

---

## Risk & Profit Management

### Risk Manager — [risk/manager.py](risk/manager.py)

| Threshold | Action |
|-----------|--------|
| **10% drawdown** | Alert + reduce all positions 50% |
| **15% drawdown** | Hard stop — go 100% cash |
| **30% drawdown** | Competition disqualification cap (never reach) |
| **40% per asset** | Max position size per token |
| **23h no trade** | Force micro-rebalance (meet 1 trade/day minimum) |
| **<1% weight change** | Skip trade (avoid gas/dust fees) |

> The 30% / 1-trade-per-day rules map directly to the competition's risk gate and
> minimum-trade requirement.

### Profit Manager — [risk/profit_manager.py](risk/profit_manager.py)

Runs *before* the model weights are applied and can override them:

| Trigger | Action |
|---------|--------|
| +10% on a position | Sell 50% (partial take-profit) |
| +20% on a position | Sell 100% (full take-profit) |
| -5% from peak | Sell 100% (trailing stop) |
| After a TP | 2-cycle (8h) cooldown before re-entry |

---

## Token Universe

```
ALL_TOKENS    = BTC BNB SOL ETH XRP INJ DOGE LTC   # trained on
TRADE_TOKENS  =     BNB SOL ETH XRP INJ DOGE LTC   # tradeable
```

**Competition caveat:** of the trained tokens, only `ETH XRP INJ DOGE LTC` are on
the official 149-token eligible list; **trades outside the list don't count**.
`BNB` and `SOL` are *not* eligible, so the agent zeroes their weights and
redistributes to eligible tokens. BSC BEP-20 contract addresses are in
[config.py:50](config.py#L50).

---

## Configuration Reference

Key knobs in [config.py](config.py):

| Setting | Default | Meaning |
|---------|---------|---------|
| `DRY_RUN` | `True` | Paper trade (log only) vs. live BSC execution |
| `INITIAL_CAPITAL` | `100.0` | Starting USD / USDT |
| `MAX_POSITION` | `0.40` | Max 40% per asset |
| `TRANSACTION_COST` | `0.001` | 10 bps modeled cost |
| `SLIPPAGE` | `0.0005` | 5 bps modeled slippage |
| `MIN_TRADE_THRESHOLD` | `0.01` | Ignore rebalances < 1% of capital |
| `CYCLE_HOURS` | `1` | Full inference cadence |
| `CHECK_INTERVAL_MIN` | `15` | Price-check / profit-manager cadence |
| `SEQUENCE_LENGTH` | `168` | 1-week feature window (must match training) |
| `X402_ENABLED` | `True` | Pay-per-call CMC Agent Hub micropayments |

Network: BSC mainnet (`chain_id 56`), PancakeSwap V2 router, CMC Pro API +
Binance public API for OHLCV fallback.

---

## Monitoring & Alerts

**Live dashboard** — [monitor.py](monitor.py):

```bash
./run.sh monitor.py                # refresh every 15s
./run.sh monitor.py --interval 5   # refresh every 5s
```

**Telegram** — [alerts/telegram.py](alerts/telegram.py) sends startup, per-trade,
risk-warning/stop, error, and daily-summary messages once `TELEGRAM_BOT_TOKEN`
and `TELEGRAM_CHAT_ID` are set.

---

## Cron Deployment

Single cycle every 4 hours:

```bash
crontab -e
# Add:
0 0,4,8,12,16,20 * * * cd /path/to/qtrader && python agent.py >> logs/cron.log 2>&1
```

For uninterrupted intraday profit management, prefer `python agent.py --loop`
under a process manager (systemd / tmux / supervisor).

---

## Paper → Live Switch

One change flips the whole stack from logging to real on-chain execution:

```python
# config.py
DRY_RUN = False   # was True
```

Same inference pipeline, same weights, same risk rules — TWAK simply signs and
broadcasts real BSC transactions instead of logging them. (Or override per-run
with `--dry-run false`.)

---

## Tech Stack

| Layer | Technology |
|-------|------------|
| Data | CoinMarketCap AI Agent Hub + x402-gated data endpoints + Binance public API |
| ML brain | Proprietary model (trained in our main architecture) |
| Decision | CMC Agent Hub skills (regime, perp/CVD analysis) |
| Execution | Trust Wallet Agent Kit (TWAK) — local signing |
| Trading chain | BNB Chain (BSC mainnet, chain id 56) |
| Payment chain | Base mainnet (chain id 8453) — real USDC x402 |
| DEX | PancakeSwap V2 |
| Agent identity | BNB AI Agent SDK (ERC-8004 / ERC-8183) |
| Payments | **Real x402 micropayments** (USDC on Base, EIP-3009 via TWAK) |
| Alerts | Telegram Bot API |

---

## Special Prize Coverage

Each special prize is $2,000 and stackable with a main placement.

| Prize | How qtrader covers it |
|-------|-----------------------|
| **Best Use of TWAK** | TWAK is the *sole* execution layer — local key signing, autonomous mode, native x402, on-chain guardrails (drawdown cap, token allowlist, per-trade/daily limits, slippage); fully self-custodial through the entire trade loop ([twak_client.py](execution/twak_client.py)) |
| **Best Use of Agent Hub / x402** | CMC Agent Hub for market data + **real x402 micropayments** (USDC on Base, EIP-3009 signed via TWAK) paying live data endpoints every cycle — verifiable on-chain, not simulated ([x402_payments.py](execution/x402_payments.py), [cmc_hub.py](data/cmc_hub.py)) |
| **Best Use of BNB Agent SDK** | Strategy registered on-chain (ERC-8004) and served as a tradeable agent endpoint (ERC-8183) ([bnb_agent.py](execution/bnb_agent.py)) |

---

## Competition Details

| Field | Value |
|-------|-------|
| Track | 1 — Autonomous Trading Agents ($24,000, 5 winners) |
| Total prize pool | $36,000 (CMC × Trust Wallet × BNB Chain) |
| Build window | June 3 – June 21, 2026 |
| On-chain registration deadline | Before June 22, 2026 |
| Live trading window | June 22 – June 28, 2026 |
| Judging | June 29 – July 5, 2026 |
| Scoring | Total return, hour-by-hour, **30% max-drawdown disqualification gate** |
| Minimum trades | 1 per day (7 over the week) |
| Eligible tokens | 149 BEP-20 tokens on CMC (ETH, XRP, DOGE, LTC, INJ in our set) |
| Competition contract | `0x212c61b9b72c95d95bf29cf032f5e5635629aed5` (BSC) |
| Agent wallet | `0x4547C17BEF404a61767DB061df14Fad7581D9aB1` |

> Returns are measured hourly; any hour starting with a portfolio worth ≤ $1
> scores 0% for that hour — so capital stays deployed for the full window.

---

## Contact

`qtrader` (this repo) is the **open, runnable trading agent**. The **research
stack behind the brain** — the model architecture, training pipeline, and how
the weights are produced — is **private and not published here**.

If you want to discuss the full research stack (judges, collaborators, or anyone
curious about the methodology), reach out directly:

**📩 ritwickhaldar2018@gmail.com**

---

**Built for BNB Chain × CoinMarketCap × Trust Wallet.**
