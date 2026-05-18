#!/usr/bin/env bash
# Run Dex header preprocessing from the only_base1_model package root.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
export PYTHONPATH="${ROOT}${PYTHONPATH:+:$PYTHONPATH}"
python -m src.preprocessing.preprocess_apks "$@"
