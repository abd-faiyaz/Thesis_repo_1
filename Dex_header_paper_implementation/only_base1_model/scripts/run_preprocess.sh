#!/usr/bin/env bash
# Run multi-Dex header preprocessing (sum-pool all classes*.dex) from package root.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
# shellcheck source=/dev/null
source "$ROOT/scripts/activate_thesis_env.sh"
export PYTHONPATH="${ROOT}${PYTHONPATH:+:$PYTHONPATH}"
"$PYTHON" -m src.preprocessing.preprocess_apks "$@"
