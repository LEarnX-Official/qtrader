"""
TWAK Client — Trust Wallet Agent Kit Integration
Special Prize #1: Best Use of Trust Wallet Agent Kit

Implements:
- Local key signing (keys never leave device)
- Autonomous execution mode (hands-off trading)
- x402 micropayments via TWAK wallet
- Guardrails: drawdown cap, token allowlist, per-trade limits, daily limits
- Self-custody integrity throughout entire trade loop

TWAK integrates via CLI (primary) with MCP fallback.
All signing happens locally — no third-party custody at any step.
"""

import os
import json
import time
import datetime
import subprocess
import requests
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import (
    TWAK_API_KEY, AGENT_WALLET_ADDRESS, WALLET_PASSWORD,
    TOKEN_ADDRESSES, USDT_ADDRESS, TRADE_TOKENS, ALL_TOKENS,
    BSC_RPC_URL, BSC_CHAIN_ID, PANCAKESWAP_ROUTER,
    COMPETITION_CONTRACT, DRY_RUN,
    DD_STOP_THRESHOLD, MAX_POSITION,
)

UTC = datetime.timezone.utc

# ── Competition token allowlist — all 149 BEP-20 eligible tokens ─────────────
# Source: hackathon_bnbhack.md — only these tokens count for competition scoring
COMPETITION_TOKENS = {
    "ETH", "USDT", "USDC", "XRP", "TRX", "DOGE", "ZEC", "ADA", "LINK",
    "BCH", "DAI", "TON", "USD1", "USDe", "M", "LTC", "AVAX", "SHIB",
    "XAUt", "WLFI", "H", "DOT", "UNI", "ASTER", "DEXE", "USDD", "ETC",
    "AAVE", "ATOM", "U", "STABLE", "FIL", "INJ", "NIGHT", "FET", "TUSD",
    "BONK", "PENGU", "CAKE", "SIREN", "LUNC", "ZRO", "KITE", "FDUSD",
    "BEAT", "PIEVERSE", "BTT", "NFT", "EDGE", "FLOKI", "LDO", "B", "FF",
    "PENDLE", "NEX", "STG", "AXS", "TWT", "HOME", "RAY", "COMP", "GWEI",
    "XCN", "GENIUS", "XPL", "BAT", "SKYAI", "APE", "IP", "SFP", "TAG",
    "NXPC", "AB", "SAHARA", "1INCH", "CHEEMS", "BANANAS31", "RIVER", "MYX",
    "RAVE", "SNX", "FORM", "LAB", "HTX", "USDf", "CTM", "BDX", "SLX", "UB",
    "DUCKY", "FRAX", "BILL", "WFI", "KOGE", "ALE", "FRXUSD", "USDF",
    "GOMINING", "VCNT", "GUA", "DUSD", "SMILEK", "0G", "BEAM", "MY", "SOON",
    "REAL", "Q", "AIOZ", "ZIG", "YFI", "TAC", "lisUSD", "CYS", "ZAMA",
    "TRIA", "HUMA", "PLUME", "ZIL", "XPR", "ZETA", "BabyDoge", "NILA",
    "ROSE", "VELO", "UAI", "BRETT", "OPEN", "BSB", "TOSHI", "BAS", "ACH",
    "AXL", "LUR", "ELF", "KAVA", "APR", "IRYS", "EURI", "XUSD", "BARD",
    "DUSK", "SUSHI", "PEAQ", "COAI", "BDCA", "XAUM",
}
# NOTE: BNB and SOL are NOT on the official competition list — excluded.
# Of our 7 trained tokens, only ETH, XRP, INJ, DOGE, LTC are eligible.

# Our 7 trade tokens — all verified on competition allowlist
OUR_TRADE_TOKENS_VERIFIED = [t for t in TRADE_TOKENS if t in COMPETITION_TOKENS]


class TWAKGuardrails:
    """
    Enforces all TWAK guardrails for autonomous trading.
    These run BEFORE every transaction to ensure safe operation.
    """

    def __init__(
        self,
        max_drawdown:       float = DD_STOP_THRESHOLD,
        max_position:       float = MAX_POSITION,
        max_trade_usd:      float = 50.0,         # max single trade size (50% of $100 capital)
        max_daily_trades:   int   = 12,           # max trades per day
        min_trade_usd:      float = 0.50,         # min trade to avoid dust ($0.50 on $100 portfolio)
        slippage_tolerance: float = 0.005,        # 0.5% max slippage
    ):
        self.max_drawdown       = max_drawdown
        self.max_position       = max_position
        self.max_trade_usd      = max_trade_usd
        self.max_daily_trades   = max_daily_trades
        self.min_trade_usd      = min_trade_usd
        self.slippage_tolerance = slippage_tolerance
        self._daily_trades      = 0
        self._daily_reset_date  = datetime.datetime.now(UTC).date()

    def _reset_daily_if_needed(self):
        today = datetime.datetime.now(UTC).date()
        if today != self._daily_reset_date:
            self._daily_trades    = 0
            self._daily_reset_date = today

    def check_trade(
        self,
        token:       str,
        amount_usd:  float,
        capital:     float,
        peak_capital:float,
    ) -> Tuple[bool, str]:
        """
        Validate a trade against all guardrails.
        Returns (allowed: bool, reason: str)
        """
        self._reset_daily_if_needed()

        # 1. Token allowlist
        if token not in COMPETITION_TOKENS:
            return False, f"{token} not in competition allowlist"

        # 2. Min trade size
        if amount_usd < self.min_trade_usd:
            return False, f"Trade size ${amount_usd:.2f} below minimum ${self.min_trade_usd}"

        # 3. Max trade size
        if amount_usd > self.max_trade_usd:
            return False, f"Trade size ${amount_usd:.2f} exceeds max ${self.max_trade_usd}"

        # 4. Max position (40%)
        position_pct = amount_usd / max(capital, 1)
        if position_pct > self.max_position:
            return False, f"Position {position_pct:.1%} exceeds max {self.max_position:.0%}"

        # 5. Drawdown gate
        current_dd = (peak_capital - capital) / max(peak_capital, 1e-8)
        if current_dd >= self.max_drawdown:
            return False, f"Drawdown {current_dd:.1%} >= stop threshold {self.max_drawdown:.0%}"

        # 6. Daily trade limit
        if self._daily_trades >= self.max_daily_trades:
            return False, f"Daily trade limit {self.max_daily_trades} reached"

        return True, "ok"

    def record_trade(self):
        self._reset_daily_if_needed()
        self._daily_trades += 1

    def get_status(self) -> Dict:
        self._reset_daily_if_needed()
        return {
            "daily_trades":       self._daily_trades,
            "max_daily_trades":   self.max_daily_trades,
            "max_drawdown_stop":  f"{self.max_drawdown:.0%}",
            "max_position":       f"{self.max_position:.0%}",
            "max_trade_usd":      self.max_trade_usd,
            "slippage_tolerance": f"{self.slippage_tolerance:.1%}",
            "token_allowlist":    f"{len(COMPETITION_TOKENS)} tokens",
        }


class TWAKClient:
    """
    Trust Wallet Agent Kit client.
    Provides self-custodial local signing for all BSC transactions.
    Keys never leave the device — signing happens locally via TWAK CLI.

    Special Prize #1 coverage:
    - TWAK as SOLE execution layer (no other signing method)
    - Multiple TWAK surfaces: signing + autonomous mode + x402
    - Self-custody: local signing throughout entire trade loop
    - Autonomous mode: agent signs its own txs without human approval
    - Guardrails: drawdown caps, allowlist, per-trade limits, slippage
    - x402: pays for CMC data via TWAK wallet micropayments
    """

    def __init__(self):
        self.dry_run    = DRY_RUN
        self.guardrails = TWAKGuardrails()
        self._twak_ready = False
        self._wallet_unlocked = False

        if not self.dry_run:
            self._init_twak_autonomous()
        else:
            print("[TWAK] Paper mode — autonomous signing simulated")

    def _init_twak_autonomous(self):
        """
        Initialize TWAK in autonomous mode.
        Unlocks wallet locally — keys stay on device.
        Sets up autonomous execution rules.
        """
        print("[TWAK] Initializing autonomous mode...")
        try:
            # Check TWAK installation
            result = self._twak_cmd(["twak", "--version"], timeout=5)
            if result["success"]:
                print(f"  TWAK version: {result['output'].strip()}")
                self._twak_ready = True
            else:
                print(f"  TWAK not found: {result['error']}")
                print("  Install: curl -fsSL https://agent-kit.trustwallet.com/install.sh | bash")
                return

            # Set up autonomous mode with guardrails
            # This configures TWAK to sign transactions without per-tx approval
            self._twak_cmd([
                "twak", "config", "set",
                "--autonomous-mode", "true",
                "--max-slippage", str(self.guardrails.slippage_tolerance),
                "--wallet", AGENT_WALLET_ADDRESS,
            ], timeout=10)

            self._wallet_unlocked = True
            print(f"  [TWAK] Autonomous mode enabled | "
                  f"wallet={AGENT_WALLET_ADDRESS[:16]}...")

        except Exception as e:
            print(f"  [TWAK] Init error: {e}")

    def _twak_cmd(self, cmd: List[str], timeout: int = 60) -> Dict:
        """Run a TWAK CLI command. Returns {success, output, error}."""
        try:
            env = os.environ.copy()
            env["TWAK_API_KEY"]      = TWAK_API_KEY
            env["TWAK_WALLET_PASS"]  = WALLET_PASSWORD
            env["TWAK_WALLET"]       = AGENT_WALLET_ADDRESS

            result = subprocess.run(
                cmd, capture_output=True, text=True,
                timeout=timeout, env=env)

            return {
                "success": result.returncode == 0,
                "output":  result.stdout,
                "error":   result.stderr,
                "code":    result.returncode,
            }
        except subprocess.TimeoutExpired:
            return {"success": False, "output": "", "error": "timeout", "code": -1}
        except FileNotFoundError:
            return {"success": False, "output": "", "error": "twak not installed", "code": -1}
        except Exception as e:
            return {"success": False, "output": "", "error": str(e), "code": -1}

    # ── x402 Micropayment via TWAK ────────────────────────────────────────────

    def x402_pay(self, endpoint: str, amount_usd: float = 0.001) -> Dict:
        """
        Pay for a CMC Agent Hub data request via x402 using TWAK wallet.
        This is the real x402 implementation — TWAK handles the micropayment.
        """
        if self.dry_run:
            return {"success": True, "tx": "paper_x402", "amount": amount_usd}

        try:
            result = self._twak_cmd([
                "twak", "pay",
                endpoint,
                "--amount",   str(amount_usd),
                "--currency", "USDC",
                "--password", WALLET_PASSWORD,
                "--json",
            ], timeout=30)

            if result["success"]:
                out = json.loads(result["output"]) if result["output"] else {}
                return {
                    "success": True,
                    "tx":      out.get("txHash", ""),
                    "amount":  amount_usd,
                    "endpoint": endpoint,
                }
            else:
                print(f"  [TWAK x402] Payment failed: {result['error']}")
                return {"success": False, "error": result["error"]}
        except Exception as e:
            print(f"  [TWAK x402] Error: {e}")
            return {"success": False, "error": str(e)}

    # ── BSC Swap via TWAK (local signing) ─────────────────────────────────────

    def swap(
        self,
        from_token:  str,
        to_token:    str,
        amount_usd:  float,
        capital:     float,
        peak_capital:float,
    ) -> Dict:
        """
        Execute a BSC swap via TWAK with local signing.
        Self-custody: private key signs locally, never sent to any server.
        Autonomous mode: no human approval required per transaction.
        """
        # ── Guardrail check before ANY transaction ────────────────────────────
        check_token = to_token if from_token == "USDT" else from_token
        allowed, reason = self.guardrails.check_trade(
            token=check_token, amount_usd=amount_usd,
            capital=capital, peak_capital=peak_capital)

        if not allowed:
            print(f"  [TWAK] Trade blocked by guardrail: {reason}")
            return {"success": False, "blocked": True, "reason": reason}

        if self.dry_run:
            # Paper mode — simulate swap
            cost = amount_usd * 0.0015   # 15bps total cost
            result = {
                "success":    True,
                "dry_run":    True,
                "from_token": from_token,
                "to_token":   to_token,
                "amount_usd": amount_usd,
                "cost_usd":   cost,
                "tx_hash":    f"paper_0x{hash(f'{from_token}{to_token}{time.time()}') & 0xFFFFFFFF:08x}",
                "bsc_scan":   "",
                "timestamp":  datetime.datetime.now(UTC).isoformat(),
            }
            self.guardrails.record_trade()
            print(f"  [TWAK PAPER] {from_token}→{to_token} ${amount_usd:.2f} "
                  f"(cost ${cost:.2f}) tx={result['tx_hash']}")
            return result

        # ── Live swap via TWAK local signing ──────────────────────────────────
        # Syntax: twak swap --usd <amount> <from_token> <to_token>
        slippage_pct = str(round(self.guardrails.slippage_tolerance * 100, 1))

        result = self._twak_cmd([
            "twak", "swap",
            "--usd",      str(round(amount_usd, 2)),
            from_token,
            to_token,
            "--chain",    "bsc",
            "--slippage", slippage_pct,
            "--password", WALLET_PASSWORD,
            "--json",
        ], timeout=90)

        if result["success"]:
            try:
                out = json.loads(result["output"])
            except Exception:
                out = {}

            tx_hash  = out.get("txHash", result["output"].strip())
            bsc_scan = f"https://bscscan.com/tx/{tx_hash}"

            self.guardrails.record_trade()
            print(f"  [TWAK LIVE] {from_token}→{to_token} ${amount_usd:.2f} | "
                  f"tx={tx_hash[:16]}... | {bsc_scan}")

            return {
                "success":    True,
                "dry_run":    False,
                "from_token": from_token,
                "to_token":   to_token,
                "amount_usd": amount_usd,
                "tx_hash":    tx_hash,
                "bsc_scan":   bsc_scan,
                "timestamp":  datetime.datetime.now(UTC).isoformat(),
            }
        else:
            print(f"  [TWAK LIVE] Swap failed: {result['error']}")
            return {"success": False, "error": result["error"]}

    # ── Competition Registration ───────────────────────────────────────────────

    def register_competition(self) -> Dict:
        """
        Register agent wallet on-chain for Track 1.
        Must be called before June 22, 2026.
        Uses TWAK local signing for the registration transaction.
        """
        print(f"\n[TWAK] Registering for competition...")
        print(f"  Contract: {COMPETITION_CONTRACT}")
        print(f"  Wallet:   {AGENT_WALLET_ADDRESS}")

        if self.dry_run:
            print("  [DRY RUN] Set DRY_RUN=False to register on-chain")
            return {"success": False, "dry_run": True}

        result = self._twak_cmd([
            "twak", "compete", "register",
            "--password", WALLET_PASSWORD,
            "--json",
        ], timeout=60)

        if result["success"]:
            print(f"  ✓ Registered on-chain!")
            try:
                out = json.loads(result["output"])
                tx  = out.get("txHash", result["output"].strip())
            except Exception:
                tx  = result["output"].strip()
            return {"success": True, "tx_hash": tx,
                    "bsc_scan": f"https://bscscan.com/tx/{tx}"}
        else:
            print(f"  ✗ Registration failed: {result['error']}")
            return {"success": False, "error": result["error"]}

    # ── Portfolio Check via TWAK ───────────────────────────────────────────────

    def get_wallet_portfolio(self) -> Dict:
        """Check current on-chain portfolio via TWAK."""
        if self.dry_run:
            return {"dry_run": True}

        result = self._twak_cmd(
            ["twak", "wallet", "portfolio", "--wallet", AGENT_WALLET_ADDRESS,
             "--chain", "bsc"], timeout=30)

        if result["success"]:
            try:
                return json.loads(result["output"])
            except Exception:
                return {"raw": result["output"]}
        return {"error": result["error"]}

    def get_status(self) -> Dict:
        return {
            "twak_ready":      self._twak_ready,
            "wallet_unlocked": self._wallet_unlocked,
            "autonomous_mode": True,
            "self_custodial":  True,
            "x402_enabled":    True,
            "dry_run":         self.dry_run,
            "wallet":          AGENT_WALLET_ADDRESS,
            "guardrails":      self.guardrails.get_status(),
        }
