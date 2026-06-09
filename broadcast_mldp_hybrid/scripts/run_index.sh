#!/usr/bin/env bash
# P1 — index APK corpus and assign train / val / test splits.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
# shellcheck source=/dev/null
source "$ROOT/scripts/activate_thesis_env.sh"
REPO_ROOT="$(cd "$ROOT/.." && pwd)"
# shellcheck source=/dev/null
source "$REPO_ROOT/scripts/thesis_pythonpath.sh"
export PYTHONPATH="${ROOT}${PYTHONPATH:+:$PYTHONPATH}"
"$PYTHON" "$ROOT/scripts/index_dataset.py" "$@"
