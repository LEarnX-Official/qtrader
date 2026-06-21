#!/usr/bin/env python3
"""
setup_bnb_wallet.py — one-time BNB Agent SDK keystore (re)import.

The bnbagent keystore in ~/.bnbagent/wallets/ for the agent wallet was
encrypted with an unknown/old password, so the BNB Agent SDK can't unlock it.
This script re-imports the wallet from its private key and re-encrypts the
keystore with the CURRENT WALLET_PASSWORD (from .env), so bnb_agent.py works.

Security:
  - The private key is read via getpass (no echo, not stored, not logged).
  - It is used only to derive + encrypt the keystore, then dropped.
  - Never pass the key as a CLI arg (it would land in shell history).

Usage:
    python setup_bnb_wallet.py
    # paste the private key for the agent wallet when prompted
"""

import sys
from pathlib import Path
from getpass import getpass

BASE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE_DIR))

from config import AGENT_WALLET_ADDRESS, WALLET_PASSWORD


def main() -> int:
    if not WALLET_PASSWORD:
        print("[ERROR] WALLET_PASSWORD is empty in .env — set it first.")
        return 1

    target = AGENT_WALLET_ADDRESS
    print("=" * 60)
    print("  BNB Agent SDK — wallet keystore (re)import")
    print("=" * 60)
    print(f"  Target wallet : {target}")
    print(f"  Password      : (from .env WALLET_PASSWORD)")
    print(f"  Keystore dir  : ~/.bnbagent/wallets/")
    print()

    # ── Remove the stale keystore (encrypted with the wrong password) ─────────
    wallets_dir = Path.home() / ".bnbagent" / "wallets"
    stale = wallets_dir / f"{target}.json"
    if stale.exists():
        backup = stale.with_suffix(".json.bak")
        stale.rename(backup)
        print(f"  Backed up stale keystore → {backup.name}")

    # ── Read the private key securely ─────────────────────────────────────────
    pk = getpass("  Paste PRIVATE KEY for the agent wallet (hidden): ").strip()
    if not pk:
        print("[ERROR] No key entered — aborting.")
        return 1

    # ── Import + encrypt with the current password ────────────────────────────
    try:
        from bnbagent import EVMWalletProvider
        wallet = EVMWalletProvider(password=WALLET_PASSWORD, private_key=pk)
    except Exception as e:
        print(f"[ERROR] Import failed: {e}")
        return 1
    finally:
        pk = None  # drop the key reference

    # ── Verify the resulting address matches the target ───────────────────────
    got = wallet.address
    if got.lower() != target.lower():
        print(f"[ERROR] Address mismatch!")
        print(f"        expected {target}")
        print(f"        got      {got}")
        print("        The keystore was created but does NOT match AGENT_WALLET_ADDRESS.")
        print("        Check you pasted the right private key.")
        return 1

    print(f"\n  ✅ Keystore created for {got}")
    print(f"     Encrypted with current WALLET_PASSWORD.")

    # ── Round-trip: load it back the way bnb_agent.py does ────────────────────
    try:
        from bnbagent import EVMWalletProvider
        EVMWalletProvider(password=WALLET_PASSWORD, address=target)
        print(f"  ✅ Verified: loads cleanly with address={target[:12]}...")
    except Exception as e:
        print(f"[WARN] Re-load check failed: {e}")
        return 1

    print("\nDone. The BNB Agent SDK can now unlock the wallet.")
    print("Test with:  python agent.py --status")
    return 0


if __name__ == "__main__":
    sys.exit(main())
