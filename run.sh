#!/bin/bash
# qtrader — run with correct venv
PYTHON="/home/devil/Desktop/thesis3/@final1/@/.venv/bin/python3"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"
exec "$PYTHON" "$@"
