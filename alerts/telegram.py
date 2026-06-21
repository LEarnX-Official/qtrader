"""
Telegram Alert System
Sends trade notifications, risk warnings, and daily summaries.
"""

import requests
import datetime
import numpy as np
from typing import Dict, Optional
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID, TRADE_TOKENS, DRY_RUN

UTC = datetime.timezone.utc


def send(message: str) -> bool:
    """Send a Telegram message. Returns True if successful."""
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        print(f"  [Telegram] (not configured) {message[:80]}")
        return False
    try:
        r = requests.post(
            f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage",
            json={"chat_id": TELEGRAM_CHAT_ID, "text": message,
                  "parse_mode": "Markdown"},
            timeout=10)
        return r.status_code == 200
    except Exception as e:
        print(f"  [Telegram] Send failed: {e}")
        return False


def alert_trade(
    weights:      np.ndarray,
    cash_w:       float,
    capital:      float,
    total_return: float,
    reason:       str,
    tx_hashes:    list = None,
):
    mode = "📋 PAPER" if DRY_RUN else "🔴 LIVE"
    now  = datetime.datetime.now(UTC).strftime("%Y-%m-%d %H:%M UTC")

    lines = [
        f"{mode} TRADE — {now}",
        f"",
        f"💰 Capital:  ${capital:,.2f}",
        f"📈 Return:   {total_return:+.2%}",
        f"💵 Cash:     {cash_w:.1%}",
        f"📋 Reason:   {reason}",
        f"",
        f"*Weights:*",
    ]
    for i, token in enumerate(TRADE_TOKENS):
        w = float(weights[i])
        if w > 0.01:
            lines.append(f"  {token}: {w:.1%}")

    if tx_hashes:
        lines.append(f"")
        lines.append(f"*Transactions:*")
        for tx in tx_hashes:
            lines.append(f"  {tx['token']}: `{tx['tx'][:20]}...`")

    send("\n".join(lines))


def alert_risk(
    level:     str,   # "warn" | "stop"
    drawdown:  float,
    capital:   float,
):
    if level == "stop":
        emoji = "🚨"
        title = "HARD STOP — Going 100% Cash"
    else:
        emoji = "⚠️"
        title = "DRAWDOWN WARNING — Reducing Positions 50%"

    now = datetime.datetime.now(UTC).strftime("%Y-%m-%d %H:%M UTC")
    msg = (f"{emoji} *{title}*\n"
           f"\n"
           f"Time:     {now}\n"
           f"Drawdown: {drawdown:.2%}\n"
           f"Capital:  ${capital:,.2f}\n"
           f"\n"
           f"{'All positions liquidated to USDT.' if level == 'stop' else 'Position sizes reduced by 50%.'}")
    send(msg)


def alert_daily_summary(
    capital:      float,
    total_return: float,
    max_dd:       float,
    n_trades:     int,
    weights:      np.ndarray,
    cash_w:       float,
):
    now  = datetime.datetime.now(UTC).strftime("%Y-%m-%d UTC")
    mode = "📋 PAPER" if DRY_RUN else "🔴 LIVE"

    lines = [
        f"📊 *Daily Summary — {now}*",
        f"Mode: {mode}",
        f"",
        f"💰 Capital:     ${capital:,.2f}",
        f"📈 Total Return: {total_return:+.2%}",
        f"📉 Max Drawdown: {max_dd:.2%}",
        f"🔄 Trades Today: {n_trades}",
        f"",
        f"*Current Portfolio:*",
        f"  Cash: {cash_w:.1%}",
    ]
    for i, token in enumerate(TRADE_TOKENS):
        w = float(weights[i])
        if w > 0.005:
            lines.append(f"  {token}: {w:.1%}")

    send("\n".join(lines))


def alert_startup(dry_run: bool, wallet: str):
    mode = "PAPER TRADE" if dry_run else "🔴 LIVE TRADE"
    now  = datetime.datetime.now(UTC).strftime("%Y-%m-%d %H:%M UTC")
    msg  = (f"🚀 *Quantum Trader Started*\n"
            f"\n"
            f"Mode:   {mode}\n"
            f"Wallet: `{wallet[:16]}...`\n"
            f"Time:   {now}\n"
            f"\n"
            f"BNB Hack: AI Trading Agent Edition\n"
            f"Running inference every 4 hours.")
    send(msg)


def alert_error(error: str):
    now = datetime.datetime.now(UTC).strftime("%Y-%m-%d %H:%M UTC")
    msg = (f"❌ *Error — {now}*\n"
           f"\n"
           f"`{error[:500]}`")
    send(msg)
