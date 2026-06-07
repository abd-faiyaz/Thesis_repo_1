#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
REPO_ROOT="$(cd "$ROOT/.." && pwd)"
cd "$ROOT"
# shellcheck source=/dev/null
source "$ROOT/scripts/activate_thesis_env.sh"
RUN_ID="${1:-${LR_RUN_ID:-}}"
if [[ -z "$RUN_ID" && -f "$ROOT/output_archives/LATEST_RUN.txt" ]]; then
  RUN_ID="$(tr -d '[:space:]' < "$ROOT/output_archives/LATEST_RUN.txt")"
fi
exec "$PYTHON" "$REPO_ROOT/scripts/thesis_run_archive.py" snippet \
  --profile linregdroid_permission --root "$ROOT" --run-id "$RUN_ID"
