#!/usr/bin/env bash
# Finalize output_archives/<run_id>/ after a Pattern A pipeline run.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REPO_ROOT="$(cd "$ROOT/../../.." && pwd)"
cd "$ROOT"

# shellcheck source=/dev/null
source "$ROOT/scripts/activate_thesis_env.sh"
export PYTHONPATH="${ROOT}${PYTHONPATH:+:$PYTHONPATH}"

RUN_ID="${1:-${PA_RUN_ID:-}}"
if [[ -z "$RUN_ID" && -f "$ROOT/output_archives/LATEST_RUN.txt" ]]; then
  RUN_ID="$(tr -d '[:space:]' < "$ROOT/output_archives/LATEST_RUN.txt")"
fi
if [[ -z "$RUN_ID" ]]; then
  echo "Usage: $0 [run_id]" >&2
  exit 1
fi

exec "$PYTHON" "$REPO_ROOT/scripts/thesis_run_archive.py" finalize \
  --profile early_fusion_dex_manifest \
  --root "$ROOT" \
  --run-id "$RUN_ID"
