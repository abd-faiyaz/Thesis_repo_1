#!/usr/bin/env bash
# Regenerate androidTest A1 parity APKs + extraction fixtures (3 samples).
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
export ROOT
cd "$ROOT"
# shellcheck source=/dev/null
source "$ROOT/scripts/activate_thesis_env.sh"
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
"$PYTHON" "$SCRIPT_DIR/generate_a1_parity_fixtures.py"
