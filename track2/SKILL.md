# CMC Strategy Skill — Regime-Gated Conviction Allocator

> **BNB Hack Track 2 — Strategy Skills** · Powered by CoinMarketCap Agent Hub
> A backtestable, LLM-authored crypto trading strategy that turns CMC Agent Hub
> data into entry/exit/sizing rules — defensive by design.

---

## What this Skill does

Given live CoinMarketCap Agent Hub data (Fear & Greed, global dominance,
per-token quotes/momentum, funding rates), this Skill outputs a **target
portfolio allocation** across a basket of crypto tokens plus cash.

Its thesis is deliberately contrarian to most momentum bots: **short-horizon
price direction is mostly unpredictable from price alone**, so the Skill leans on
*regime* and *positioning* data and **defaults to cash when conviction is low**.
This is what lets it survive drawdown-capped evaluation while still capturing
trend when the regime is constructive.

It composes three sub-signals (each a usable Skill on its own):

1. **Regime gate** — Fear & Greed + BTC dominance → an overall *risk budget*
   (how much of the portfolio may be deployed at all).
2. **Momentum conviction** — blended 1h/24h/7d returns per token → directional
   conviction, but only acted on inside the regime budget.
3. **Sentiment-divergence filter** — funding rates vs price action → flags
   crowded/abandoned positioning to avoid buying euphoria and selling capitulation.

---

## Inputs (all from CMC Agent Hub)

| Input | CMC source | Used for |
|-------|-----------|----------|
| Fear & Greed index | `/v3/fear-and-greed` | Regime gate, contrarian sizing |
| Global metrics (BTC dominance, total mcap) | `/v1/global-metrics/quotes` | Regime gate |
| Per-token quotes (1h/24h/7d %, volume) | `/v2/cryptocurrency/quotes` | Momentum conviction |
| Funding rates (per token) | derivatives feed | Sentiment-divergence filter |

**Universe:** BNB, SOL, ETH, XRP, INJ, DOGE, LTC (+ CASH).

---

## Strategy logic (entry / exit / sizing rules)

### Step 1 — Regime gate → risk budget `R ∈ [0,1]`

```
Fear & Greed value (FG)         Risk budget R
  FG < 20  (extreme fear)         1.00   ← contrarian: max deploy
  20 ≤ FG < 35 (fear)             0.70
  35 ≤ FG < 55 (neutral)          0.50
  55 ≤ FG < 70 (greed)            0.30
  FG ≥ 70  (extreme greed)        0.10   ← de-risk, mostly cash
```

Modifier: if BTC dominance is **rising fast** (alt risk-off), multiply `R` by
0.8 (capital rotates to BTC/cash, not alts).

`CASH weight = 1 − R` (always hold at least `1 − R` in cash).

### Step 2 — Momentum conviction per token

```
mom = 0.5·(1h %) + 0.3·(24h %) + 0.2·(7d %)

raw_score(token) = max(0, mom)        # long-only; negatives → 0 (to cash)
```

### Step 3 — Sentiment-divergence filter

```
If funding_rate(token) is very positive AND price already up strongly
   → crowded long → halve raw_score  (avoid buying euphoria)
If funding_rate(token) is very negative AND price is down
   → capitulation → boost raw_score ×1.25  (contrarian entry)
```

### Step 4 — Allocate

```
deployable = R                                  # from Step 1
weights = softmax_or_normalize(raw_score) · deployable
# zero any weight < 5% → fold into CASH (avoid dust)
CASH = 1 − sum(weights)
```

### Step 5 — Exit / rebalance rules

- **Take-profit:** trim a position by 50% at +10% unrealized, 100% at +20%.
- **Trailing stop:** exit a position fully if it falls 5% from its peak.
- **Regime exit:** if FG crosses into extreme greed (≥70), cut all risk budget
  to 0.10 next cycle (force de-risk).
- **No-conviction default:** if every `raw_score` is 0 → **100% CASH**.

---

## Why it wins (Track 2 judging criteria)

- **Technical execution** — composes three distinct CMC data types (sentiment,
  positioning, momentum) into one coherent, backtestable rule set; not a single
  indicator.
- **Originality** — *defensive-first*: it treats "do nothing / hold cash" as a
  first-class action, the opposite of typical momentum skills. Honest about
  OHLCV unpredictability and designs around it.
- **Real-world relevance** — directly survivable under a max-drawdown gate; a
  self-custody user could run it unattended.
- **Demo** — fully backtestable from the spec in [BACKTEST.md](BACKTEST.md);
  reference implementation in [skill_strategy.py](skill_strategy.py).

---

## Files

| File | What |
|------|------|
| [SKILL.md](SKILL.md) | This spec — the strategy as an LLM-authored Skill |
| [skill_strategy.py](skill_strategy.py) | Reference implementation (pure function: data → weights) |
| [BACKTEST.md](BACKTEST.md) | Backtest methodology, metrics, and how to reproduce |

**Contact (research methodology):** chanakyaa0.2.0@gmail.com
