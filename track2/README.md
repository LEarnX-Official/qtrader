# BNB Hack — Track 2 Strategy Skill

**Regime-Gated Conviction Allocator** — a CMC Agent Hub Strategy Skill that turns
live market data into a defensive, backtestable crypto allocation strategy.

> Track 2 — Strategy Skills · $6,000 / 3 winners · Powered by CoinMarketCap
> Deliverable: a backtestable strategy spec authored as an LLM Skill (no
> execution layer required).

## TL;DR

A long-only + cash crypto strategy that:
- **Gates risk by market regime** (Fear & Greed + BTC dominance) — holds cash
  when conviction is low.
- **Sizes by multi-horizon momentum**, but **fades crowded positioning** using
  funding-rate divergence.
- Defaults to **100% cash** when nothing has conviction — built to survive a
  max-drawdown gate.

Backtest (Jan–May 2026, unseen): **+70% return, 2.85 Sharpe, 5.87% max DD** vs an
equal-weight benchmark that lost money.

## Files

| File | What |
|------|------|
| [SKILL.md](SKILL.md) | The strategy as an LLM-authored Skill spec (rules) |
| [skill_strategy.py](skill_strategy.py) | Reference implementation — pure `data → weights` function |
| [BACKTEST.md](BACKTEST.md) | Backtest methodology, metrics, reproduction |

## Run it

```bash
python skill_strategy.py     # prints a sample allocation from CMC-style inputs
```

## Relationship to our Track 1 agent

This Skill is the **transparent, rules-based distillation** of the decision logic
used by our Track 1 autonomous agent (`qtrader`). Track 1 runs the full
proprietary ML model; Track 2 presents the *strategy* in a readable, auditable,
backtestable form — exactly what the Strategy Skills track asks for.

**Contact:** chanakyaa0.2.0@gmail.com
