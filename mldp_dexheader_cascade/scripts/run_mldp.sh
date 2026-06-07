#!/usr/bin/env bash
# P2a — MLDP selection runs inside preprocess (PRNR→SPR→PMAR on train split).
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")/.." && pwd)"
cd "$ROOT"
# shellcheck source=/dev/null
source "$ROOT/scripts/activate_thesis_env.sh"
echo "MLDP selection is run inside preprocess_apks.py (PRNR→SPR→PMAR, train split only)."
echo "Use: bash scripts/run_preprocess.sh"
exec bash "$ROOT/scripts/run_preprocess.sh" "$@"
