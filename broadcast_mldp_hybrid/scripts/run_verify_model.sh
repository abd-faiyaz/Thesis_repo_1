#!/usr/bin/env bash
# P4 — verify deployment models and paper sklearn baselines.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
# shellcheck source=/dev/null
source "$ROOT/scripts/activate_thesis_env.sh"
export PYTHONPATH="${ROOT}${PYTHONPATH:+:$PYTHONPATH}"
"$PYTHON" "$ROOT/scripts/verify_model.py" "$@"
