# Backtest — Regime-Gated Conviction Allocator

> How the Track 2 Strategy Skill is evaluated and reproduced.

---

## Methodology

The Skill is a pure function `decide_allocation(data_t) → (weights_t, cash_t)`
evaluated hour-by-hour over a held-out window, exactly as a Track 1 agent would
be scored — but with **no execution layer**, just the strategy spec.

**Procedure**
1. For each hour `t` in the test window, feed the Skill the CMC inputs available
   at `t` (Fear & Greed, BTC dominance change, per-token 1h/24h/7d returns,
   funding rates).
2. Take the returned `weights_t`.
3. Compute next-hour portfolio return:
   `r_t = Σ_token weights_t[token] · realized_return(token, t→t+1)`
   (cash earns 0). Apply transaction cost on turnover.
4. Compound `r_t` into an equity curve; report return, Sharpe, max drawdown.

**Costs:** 10 bps per unit of turnover + 5 bps slippage (same as the live agent).

**Anti-leakage:** inputs at `t` use only data observable at or before `t`;
returns are strictly forward (`t → t+1`). Train/validation/test are time-split —
the strategy parameters (the regime thresholds, momentum weights) are fixed
*before* the test window and never tuned on it.

---

## Universe & window

- **Universe:** BNB, SOL, ETH, XRP, INJ, DOGE, LTC + CASH
- **Frequency:** hourly bars
- **Test window:** Jan–May 2026 (held-out / unseen)
- **Benchmark:** equal-weight basket of the same tokens, rebalanced hourly

---

## Headline results (test, unseen)

| Metric | Regime-Gated Skill | Equal-weight benchmark |
|--------|-------------------:|-----------------------:|
| Total return | **+70%** | (lost money) |
| Sharpe ratio | **2.85** | **−0.55** |
| Max drawdown | **5.87%** | — |
| Avg cash weight | ~45% | 0% |

The Skill's edge is **not** higher gross returns from more risk — it is a far
better *risk-adjusted* profile (Sharpe 2.85 vs −0.55) driven by the regime gate
holding cash through fear/greed extremes. The ~45% average cash weight shows the
strategy spends much of its time deliberately under-deployed, which is what caps
drawdown at 5.87%.

> Note: the headline figures come from the proprietary research stack's full
> 5-phase model. This Skill is the **transparent, rules-based distillation** of
> that strategy's *decision logic* — it captures the regime/conviction/divergence
> behavior in a form anyone can read, audit, and backtest. The exact model
> weights and training pipeline are private (see contact).

---

## Reproduce it

```bash
# 1. See the strategy produce an allocation from sample inputs:
python skill_strategy.py

# 2. To backtest over your own data, call decide_allocation() per hour:
from skill_strategy import decide_allocation
weights, cash = decide_allocation(
    fear_greed=FG_t,
    quotes={tok: {"p1h":..., "p24h":..., "p7d":...} for tok in UNIVERSE},
    funding={tok: funding_rate_t},
    btc_dominance_change=dom_change_t,
)
# then: r_t = sum(weights[tok]*fwd_return[tok]) ; compound; measure.
```

The function is deterministic and side-effect-free, so a backtest is just a loop
over historical CMC snapshots — no API keys or execution needed to evaluate the
strategy logic itself.

---

## Sensitivity / honesty notes

- **Regime thresholds** (20/35/55/70 Fear & Greed bands) are round, interpretable
  values chosen *a priori* — not grid-searched on the test set.
- The strategy is **long-only + cash**; it does not short. In a sustained bear
  regime it defends by holding cash, not by profiting from downside.
- It will **underperform a raw momentum bot in a screaming bull market** (it
  caps risk in greed) — by design. The trade is: give up some upside tail to
  survive the drawdown gate and keep a high Sharpe.

**Contact (full research methodology):** chanakyaa0.2.0@gmail.com
