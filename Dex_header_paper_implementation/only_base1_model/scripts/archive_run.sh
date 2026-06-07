#!/usr/bin/env bash
# Finalize output_archives/<run_id>/ after a BM1 pipeline run (checksums + manifest patch).
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

RUN_ID="${1:-}"
if [[ -z "$RUN_ID" && -f "$ROOT/output_archives/LATEST_RUN.txt" ]]; then
  RUN_ID="$(tr -d '[:space:]' < "$ROOT/output_archives/LATEST_RUN.txt")"
fi
if [[ -z "$RUN_ID" ]]; then
  echo "Usage: $0 [run_id]" >&2
  echo "  or set output_archives/LATEST_RUN.txt" >&2
  exit 1
fi

ARCHIVE="$ROOT/output_archives/$RUN_ID"
if [[ ! -d "$ARCHIVE" ]]; then
  echo "ERROR: archive dir not found: $ARCHIVE" >&2
  exit 1
fi

# shellcheck source=/dev/null
source "$ROOT/scripts/activate_thesis_env.sh"
export PYTHONPATH="${ROOT}${PYTHONPATH:+:$PYTHONPATH}"

shift  # run_id already consumed above; pass only extra flags (e.g. --apk-root)
"$PYTHON" "$ROOT/scripts/finalize_bm1_archive.py" --run-id "$RUN_ID" "$@"
