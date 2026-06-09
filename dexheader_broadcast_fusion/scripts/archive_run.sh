#!/usr/bin/env bash
# Finalize output_archives/<run_id>/ after a dexheader_broadcast_fusion pipeline run.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REPO_ROOT="$(cd "$ROOT/.." && pwd)"
cd "$ROOT"

# shellcheck source=/dev/null
source "$ROOT/scripts/activate_thesis_env.sh"
export PYTHONPATH="${ROOT}${PYTHONPATH:+:$PYTHONPATH}"

RUN_ID="${1:-${DBF_RUN_ID:-}}"
if [[ -z "$RUN_ID" && -f "$ROOT/output_archives/LATEST_RUN.txt" ]]; then
  RUN_ID="$(tr -d '[:space:]' < "$ROOT/output_archives/LATEST_RUN.txt")"
fi
if [[ -z "$RUN_ID" ]]; then
  echo "Usage: $0 [run_id]" >&2
  echo "  or set output_archives/LATEST_RUN.txt" >&2
  exit 1
fi

exec "$PYTHON" "$REPO_ROOT/scripts/thesis_run_archive.py" finalize \
  --profile dexheader_broadcast_fusion \
  --root "$ROOT" \
  --run-id "$RUN_ID"
