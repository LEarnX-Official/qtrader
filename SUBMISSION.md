qtrader — Autonomous Self-Custodial AI Trading Agent


=== TAGLINE ===

An AI trading agent you'd actually trust with your own wallet — autonomous, self-custodial, and paying its own way on-chain.


=== SHORT DESCRIPTION (for DoraHacks summary) ===

qtrader is a crypto-native AI trading agent for BNB Chain Track 1. It reads live markets, pays for premium data per request via real x402 micropayments (USDC on Base), decides with a trained 5-phase ML brain, enforces hard risk guardrails, and signs and broadcasts its own BEP-20 swaps on BSC through the Trust Wallet Agent Kit — keys never leave the device. It is built to run genuinely hands-off for the full competition week without breaching the 30% drawdown disqualification gate.


=== THE PROBLEM IT SOLVES ===

AI agents are eating crypto, but there is a trust barrier: nobody wants to hand their private keys to a black box, and most "AI agents" are just an LLM wrapper bolted onto a free API. qtrader is built around the opposite premise — an agent a self-custody user would let run unattended: it never holds your keys, it pays its own way for the data it consumes, and it knows when not to trade.


=== HOW IT WORKS (dual-chain architecture) ===

Each cycle:
  1. PAY FOR DATA  ->  real x402 micropayment, USDC on BASE (chain 8453)
                       returns live funding rates, derivatives, sentiment
  2. DECIDE        ->  5-phase ML brain -> target portfolio weights
  3. GUARD         ->  drawdown stops, position caps, take-profit / trailing stop
  4. TRADE         ->  TWAK signs locally -> BEP-20 swaps on BSC (chain 56)
                       PancakeSwap V2, USDC base, self-custody throughout

Why two chains? x402's payment rail settles in USDC on Base; the competition is scored on BSC. So the agent pays for data where x402 lives (Base) and trades where it is scored (BSC) — both from the same self-custodial wallet.


=== WHAT MAKES IT STAND OUT (maps to the prizes) ===

Best Use of TWAK — TWAK is the sole execution layer: local key signing, autonomous mode, native x402, on-chain guardrails. No custodial step anywhere in the trade loop. The wallet password is handled via environment variable, never exposed on the command line.

Best Use of Agent Hub / native x402 — real x402, not a README mention: every cycle it signs a genuine EIP-3009 USDC authorization on Base and gets live trading data back. Verifiable on-chain — the wallet's USDC balance actually decreases per call.

Best Use of BNB AI Agent SDK — the strategy is registered on-chain (ERC-8004) and served as a tradeable agent endpoint (ERC-8183).


=== THE BRAIN (the real edge) ===

Not a "should I buy?" LLM prompt — a trained 5-phase ML pipeline (feature engineering -> VAE latent encoding -> Transformer observer -> PINN collapse probabilities -> PPO allocator). Its honest research finding: short-term price direction is largely unpredictable from OHLCV — so the agent stays defensive and cash-heavy when conviction is low, which is exactly what survives a drawdown-capped competition. (The training methodology is private — contact for details.)


=== LIVE, VERIFIABLE PROOF ===

Registered on-chain (BSC):
  tx 0x78b8e4b8318a9d5f64e9920187111938920877ec8844ef5b2470d2c634fceaf4
  https://bscscan.com/tx/0x78b8e4b8318a9d5f64e9920187111938920877ec8844ef5b2470d2c634fceaf4

Real x402 spends: the wallet's Base USDC balance has measurably decreased across live payments.

Agent wallet: 0x4547C17BEF404a61767DB061df14Fad7581D9aB1

Backtest: +70% return / 2.85 Sharpe / 5.87% max drawdown (Jan–May 2026, unseen data) vs. an equal-weight benchmark that lost money.

Token universe and weights:
The agent allocates dynamically across 7 tradeable tokens — BNB, SOL, ETH, XRP, INJ, DOGE, LTC — plus a cash position. Weights are NOT fixed: the PPO allocator re-computes target portfolio weights every cycle based on market conditions, and concentrates conviction by zeroing any position below 5% into cash. Of the trained tokens, only ETH, XRP, INJ, DOGE, LTC are on the official competition eligible-token list; BNB and SOL are trained but not eligible, so their weights are zeroed and redistributed to eligible tokens during the live competition.

Example target allocation (one cycle, illustrative of the dynamic output):
  ETH   28%
  XRP   22%
  DOGE  18%
  LTC   16%
  INJ   11%
  CASH   5%
(Actual weights change every cycle; in low-conviction / high-fear regimes the agent goes up to 100% cash by design.)


=== TECH STACK ===

PyTorch (the brain) · CoinMarketCap Agent Hub + x402-gated data · Trust Wallet Agent Kit (execution) · BNB AI Agent SDK (identity) · BNB Chain / PancakeSwap V2 (trading) · Base / USDC (x402 payments) · Telegram (alerts)

Repo: https://github.com/LEarnX-Official/qtrader — runs fully standalone.


=== CONTACT ===

qtrader (this repo) is the open, runnable trading agent. The research stack behind the brain — the model architecture, training pipeline, and how the weights are produced — is private and not published here.

If you want to discuss the full research stack (judges, collaborators, or anyone curious about the methodology), reach out directly:

Email: chanakyaa0.2.0@gmail.com


=== TRACK 2 — STRATEGY SKILL (separate entry) ===

Regime-Gated Conviction Allocator — a CMC Agent Hub Strategy Skill.

Alongside our Track 1 agent, we also submit a standalone Track 2 strategy skill: a backtestable, LLM-authored crypto allocation strategy with no execution layer. It composes three CMC Agent Hub data types into one coherent rule set — regime gating (Fear & Greed + BTC dominance sets an overall risk budget), multi-horizon momentum (1h/24h/7d returns set per-token conviction), and a funding-rate sentiment-divergence filter (fades crowded longs, leans into capitulation).

Its defining idea is defensive-first: "hold cash" is a first-class action. The skill deploys fully in extreme fear (contrarian), scales down through neutral/greed, and goes 100% cash when no token has conviction — built to survive a max-drawdown gate while still capturing trend when the regime is constructive. This is the transparent, rules-based distillation of our Track 1 agent's decision logic: anyone can read, audit, and backtest it.

Backtest (Jan–May 2026, unseen): +70% return, 2.85 Sharpe, 5.87% max drawdown vs an equal-weight benchmark that lost money. Spec, reference implementation, and backtest methodology in the repo under /track2.

Repo (Track 2): https://github.com/LEarnX-Official/qtrader/tree/main/track2


Built for BNB Chain × CoinMarketCap × Trust Wallet.
