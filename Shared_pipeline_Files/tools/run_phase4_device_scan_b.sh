#!/usr/bin/env bash
# Phase 4 — Cascade (deployed) scan on POCO, pull, validate, compare with Scan A.
#
# Prerequisites:
#   - ./Android_Works/run_all_a4_gates.sh (A4 parity gates green)
#   - Scan A already pulled to scan_a_all_models/ (Phase 3)
#
#   ./run_phase4_device_scan_b.sh
#   # On POCO: Cascade ON → Clear scan history → Rescan all
#   ./run_phase4_device_scan_b.sh --pull-only --min-scans 400
#
# Options:
#   --pull-only       Only pull + filter + validate + compare
#   --min-scans N     Minimum cascade scans (default 1; thesis: 400)
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
SCAN_SUBDIR="scan_b_cascade"
SCAN_A_DIR="${REPO_ROOT}/Shared_pipeline_Files/results/device/scan_a_all_models"
DEST="${REPO_ROOT}/Shared_pipeline_Files/results/device/${SCAN_SUBDIR}"

PULL_ONLY=0
MIN_SCANS=1

while [[ $# -gt 0 ]]; do
  case "$1" in
    --pull-only) PULL_ONLY=1 ;;
    --min-scans) MIN_SCANS="${2:?}"; shift ;;
    -h|--help)
      sed -n '2,18p' "$0"
      exit 0
      ;;
    *)
      echo "Unknown option: $1" >&2
      exit 2
      ;;
  esac
  shift
done

if [[ "$PULL_ONLY" -eq 0 ]]; then
  cat <<EOF
=== Phase 4 — Scan B (Cascade deployed) ===

Optional before Scan B (only clears cascade file; Scan A log is kept):
  $SCRIPT_DIR/clear_device_metrics.sh cascade

On POCO F3 (no reinstall needed):
1. Toggle Scan mode → "Cascade (deployed)" (cascade ON).
2. Tap "Clear scan history" (dedup store — required to rescan same APKs).
3. Tap "Rescan all" on the eval manifest.
4. Wait until the scan queue finishes.

Then pull + validate:
  $0 --pull-only --min-scans 400

EOF
  exit 0
fi

if ! adb get-state >/dev/null 2>&1; then
  echo "adb device not connected — cannot pull metrics." >&2
  exit 1
fi

echo "=== Pulling device metrics → ${SCAN_SUBDIR}/ ==="
"$SCRIPT_DIR/pull_device_metrics.sh" "$SCAN_SUBDIR"

METRICS=""
if [[ -f "$DEST/scan_b_cascade.jsonl" ]]; then
  METRICS="$DEST/scan_b_cascade.jsonl"
elif [[ -f "$DEST/scan_b_cascade.json" ]]; then
  METRICS="$DEST/scan_b_cascade.json"
elif [[ -f "$DEST/all_scan_metrics.jsonl" ]]; then
  METRICS="$DEST/all_scan_metrics.jsonl"
elif [[ -f "$DEST/all_scan_metrics.json" ]]; then
  METRICS="$DEST/all_scan_metrics.json"
else
  echo "No metrics file under $DEST" >&2
  exit 1
fi

SCAN_A_ARG=()
if [[ -f "$SCAN_A_DIR/scan_a_all_models.jsonl" ]]; then
  SCAN_A_ARG=(--scan-a "$SCAN_A_DIR/scan_a_all_models.jsonl")
elif [[ -f "$SCAN_A_DIR/scan_a_all_models.json" ]]; then
  SCAN_A_ARG=(--scan-a "$SCAN_A_DIR/scan_a_all_models.json")
elif [[ -f "$SCAN_A_DIR/all_scan_metrics.jsonl" ]]; then
  SCAN_A_ARG=(--scan-a "$SCAN_A_DIR/all_scan_metrics.jsonl")
elif [[ -f "$SCAN_A_DIR/all_scan_metrics.json" ]]; then
  SCAN_A_ARG=(--scan-a "$SCAN_A_DIR/all_scan_metrics.json")
fi

echo "=== Validating Scan B ==="
python3 "$SCRIPT_DIR/validate_scan_b.py" "$METRICS" \
  --min-scans "$MIN_SCANS" \
  "${SCAN_A_ARG[@]}" \
  --write-report "$DEST/cascade_device_report.json"

echo "=== Comparing cascade vs ablation ==="
COMPARE_ARGS=("$METRICS")
if [[ ${#SCAN_A_ARG[@]} -gt 0 ]]; then
  COMPARE_ARGS+=("${SCAN_A_ARG[@]}")
fi
python3 "$SCRIPT_DIR/compare_cascade_eval.py" \
  "${COMPARE_ARGS[@]}" \
  --write-report "$DEST/cascade_comparison_report.json"
