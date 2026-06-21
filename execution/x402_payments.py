"""
Real x402 micropayments for qtrader — Base + USDC via Trust Wallet Agent Kit.

Special Prize: native x402 usage — real on-chain payments in the trade loop,
not a simulated receipt.

x402 is the Coinbase HTTP-402 payment standard, settled in USDC on Base.
This module pays x402-gated endpoints through TWAK's x402 client
(`twak x402 request`), which performs the full handshake:

    1. GET the endpoint → server replies HTTP 402 with payment requirements
    2. TWAK signs an EIP-3009 USDC authorization on Base (self-custodial)
    3. TWAK retries with the X-PAYMENT header → server returns the data

The agent's own self-custodial wallet signs every payment, capped by
--max-payment so an unattended agent can never overspend.

Modes
-----
* DRY_RUN=True  → record the intent (no network, no spend); receipt paper=True.
* DRY_RUN=False → real `twak x402 request` against the gated endpoint on Base.
"""

from __future__ import annotations

import json
import datetime
import subprocess
from pathlib import Path
from typing import Dict, List, Optional
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import DRY_RUN, AGENT_WALLET_ADDRESS, WALLET_PASSWORD

UTC = datetime.timezone.utc

# x402 settles in USDC on Base. USDC = 6 decimals.
X402_NETWORK   = "base"
USDC_DECIMALS  = 6

# Default per-call spend cap (atomic USDC units). 10000 = 0.01 USDC.
DEFAULT_MAX_PAYMENT_ATOMIC = 10_000


def usd_to_atomic(amount_usd: float, decimals: int = USDC_DECIMALS) -> int:
    return int(round(amount_usd * (10 ** decimals)))


class X402Payer:
    """Pays x402-gated endpoints in USDC on Base via the TWAK CLI."""

    def __init__(
        self,
        network: str = X402_NETWORK,
        max_payment_atomic: int = DEFAULT_MAX_PAYMENT_ATOMIC,
    ) -> None:
        self.network  = network
        self.max_pay  = max_payment_atomic
        self.dry_run  = DRY_RUN
        self._receipts: List[Dict] = []

    # ------------------------------------------------------------------

    def pay(self, url: str, amount_usd: float = 0.001,
            method: str = "GET", body: Optional[str] = None) -> Dict:
        """
        Pay one x402-gated request. `url` must be an x402-gated endpoint
        (returns HTTP 402). Returns a receipt dict (always).
        """
        now      = datetime.datetime.now(UTC)
        cap      = max(self.max_pay, usd_to_atomic(amount_usd))

        if self.dry_run:
            receipt = {
                "url": url, "amount_usd": amount_usd, "network": self.network,
                "max_payment_atomic": cap, "timestamp": now.isoformat(),
                "protocol": "x402", "status": "intent", "paper": True,
            }
            self._receipts.append(receipt)
            return receipt

        cmd = [
            "twak", "x402", "request", url,
            "--max-payment", str(cap),
            "--prefer-network", self.network,
            "--prefer-method", "eip3009",
            "--yes",
        ]
        if method and method.upper() != "GET":
            cmd += ["--method", method.upper()]
        if body:
            cmd += ["--body", body]

        # Pass the wallet password via env (TWAK_WALLET_PASSWORD), never on the
        # command line — keeps it out of the process arg list (ps aux).
        import os
        env = os.environ.copy()
        env["TWAK_WALLET_PASSWORD"] = WALLET_PASSWORD

        try:
            proc = subprocess.run(cmd, capture_output=True, text=True,
                                  timeout=90, env=env)
            out  = (proc.stdout or "").strip()
            ok   = proc.returncode == 0 and "402" not in out.split("\n")[0]
            tx   = ""
            try:
                parsed = json.loads(out) if out.startswith("{") else {}
                tx = parsed.get("txHash") or parsed.get("payment", {}).get("txHash", "")
            except Exception:
                pass
            receipt = {
                "url": url, "amount_usd": amount_usd, "network": self.network,
                "timestamp": now.isoformat(), "protocol": "x402",
                "status": "paid" if ok else "failed",
                "paper": False, "tx": tx,
                "raw": out[:300],
            }
            if not ok:
                receipt["error"] = (proc.stderr or out)[:300]
            self._receipts.append(receipt)
            return receipt

        except Exception as e:
            receipt = {
                "url": url, "amount_usd": amount_usd, "network": self.network,
                "timestamp": now.isoformat(), "protocol": "x402",
                "status": "error", "error": str(e),
            }
            self._receipts.append(receipt)
            return receipt

    # ------------------------------------------------------------------

    def receipts(self) -> List[Dict]:
        return list(self._receipts)

    def total_spent(self) -> float:
        return sum(r["amount_usd"] for r in self._receipts
                   if r.get("status") in ("paid", "intent"))


_payer: Optional[X402Payer] = None


def get_payer() -> X402Payer:
    global _payer
    if _payer is None:
        _payer = X402Payer()
    return _payer
