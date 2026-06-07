#!/usr/bin/env bash
# Regenerate androidTest parity manifests + extraction fixtures for A4.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
REPO_ROOT="$(cd "$ROOT/.." && pwd)"

if [[ -f "$REPO_ROOT/thesis_venv/bin/activate" ]]; then
  # shellcheck source=/dev/null
  source "$REPO_ROOT/thesis_venv/bin/activate"
fi
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"

python3 "$SCRIPT_DIR/generate_a4_parity_fixtures.py"
