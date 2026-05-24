#!/usr/bin/env bash
# Train Pattern B DualBranchNet on cached shards (Phase 5).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
# shellcheck source=/dev/null
source "$ROOT/scripts/activate_thesis_env.sh"
export PYTHONPATH="${ROOT}${PYTHONPATH:+:$PYTHONPATH}"
"$PYTHON" -m src.training.train "$@"
