#!/usr/bin/env bash
# P5 — train Mode A + Stage 1, ablations, and paper SVM/DT baselines.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
# shellcheck source=/dev/null
source "$ROOT/scripts/activate_thesis_env.sh"
export PYTHONPATH="${ROOT}${PYTHONPATH:+:$PYTHONPATH}"

EXTRA=()
if [[ "${QUICK:-0}" == "1" ]]; then
  EXTRA+=(--epochs 8 --skip-baselines)
  echo "QUICK=1 → 8 epochs, skipping sklearn baselines"
fi

"$PYTHON" -m src.training.train "${EXTRA[@]}" "$@"
