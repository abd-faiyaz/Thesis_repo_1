#!/usr/bin/env bash
# Activate shared thesis venv at repo root.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
REPO_ROOT="$(cd "$ROOT/.." && pwd)"
VENV="$REPO_ROOT/thesis_venv"

if [[ -f "$VENV/bin/activate" ]]; then
  # shellcheck source=/dev/null
  source "$VENV/bin/activate"
else
  echo "Warning: thesis_venv not found at $VENV — using current Python." >&2
fi
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
