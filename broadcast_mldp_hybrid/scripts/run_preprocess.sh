#!/usr/bin/env bash
# P2 — manifest parse, MLDP freeze S, receiver vocab A, vectorize splits.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
# shellcheck source=/dev/null
source "$ROOT/scripts/activate_thesis_env.sh"
export PYTHONPATH="${ROOT}${PYTHONPATH:+:$PYTHONPATH}"
"$PYTHON" -m src.preprocessing.preprocess_apks "$@"
