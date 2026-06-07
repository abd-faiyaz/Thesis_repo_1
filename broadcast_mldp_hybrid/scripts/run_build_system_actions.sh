#!/usr/bin/env bash
# P0a — compile assets/system_actions.json (broadcast receiver allow-list).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
# shellcheck source=/dev/null
source "$ROOT/scripts/activate_thesis_env.sh"
export PYTHONPATH="${ROOT}${PYTHONPATH:+:$PYTHONPATH}"
"$PYTHON" "$ROOT/scripts/build_system_actions.py" "$@"
