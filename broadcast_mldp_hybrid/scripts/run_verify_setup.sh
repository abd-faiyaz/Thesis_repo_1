#!/usr/bin/env bash
# P0b — verify environment, config, corpus, and system_actions.json.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
# shellcheck source=/dev/null
source "$ROOT/scripts/activate_thesis_env.sh"
export PYTHONPATH="${ROOT}${PYTHONPATH:+:$PYTHONPATH}"
"$PYTHON" "$ROOT/scripts/verify_setup.py" "$@"
