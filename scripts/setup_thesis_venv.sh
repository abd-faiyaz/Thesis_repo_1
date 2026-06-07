#!/usr/bin/env bash
# One-time setup: create thesis_venv at repo root and install all pipeline deps.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

VENV_DIR="${THESIS_VENV:-$REPO_ROOT/thesis_venv}"
REQS="${REPO_ROOT}/requirements-thesis-all.txt"

if [[ ! -f "$REQS" ]]; then
  echo "ERROR: missing $REQS" >&2
  exit 1
fi

if [[ ! -x "$VENV_DIR/bin/python" ]]; then
  echo "Creating venv: $VENV_DIR"
  python3 -m venv "$VENV_DIR"
fi

echo "Installing dependencies into $VENV_DIR ..."
"$VENV_DIR/bin/python" -m pip install -U pip
"$VENV_DIR/bin/python" -m pip install -r "$REQS"

echo ""
echo "Done. thesis_venv is ready for BM1, Pattern A/B, LinRegDroid, and MLDP."
echo "  export PYTHON=$VENV_DIR/bin/python"
echo "  # or rely on auto-detect in run_*.sh scripts"
