#!/usr/bin/env bash
# Generate THESIS_SNIPPET.md for a Pattern B archive run.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
REPO_ROOT="$(cd "$ROOT/../../.." && pwd)"
cd "$ROOT"
# shellcheck source=/dev/null
source "$ROOT/scripts/activate_thesis_env.sh"
RUN_ID="${1:-${PB_RUN_ID:-}}"
if [[ -z "$RUN_ID" && -f "$ROOT/output_archives/LATEST_RUN.txt" ]]; then
  RUN_ID="$(tr -d '[:space:]' < "$ROOT/output_archives/LATEST_RUN.txt")"
fi
exec "$PYTHON" "$REPO_ROOT/scripts/thesis_run_archive.py" snippet \
  --profile dual_branch_dex_manifest \
  --root "$ROOT" \
  --run-id "$RUN_ID"
