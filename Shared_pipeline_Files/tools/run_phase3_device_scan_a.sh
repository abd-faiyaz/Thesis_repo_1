#!/usr/bin/env bash
# Phase 3 — stage models, build/install APK, optional push, pull, validate Scan A.
#
# Prerequisite: ./Android_Works/run_all_a4_gates.sh (all A4 parity gates green).
#
# Typical thesis run (phone connected, eval APKs already pushed):
#   ./run_phase3_device_scan_a.sh
#   # On POCO: Ablation mode → Rescan all → wait for completion
#   ./run_phase3_device_scan_a.sh --pull-only --min-scans 400
#
# Options:
#   --skip-stage      Do not run stage_all_models.sh
#   --skip-build      Do not gradlew installDebug
#   --skip-push       Do not push_device_eval_apks.sh
#   --pull-only       Skip stage/build/push; only pull + validate
#   --min-scans N     Passed to validate_scan_a.py (default 1)
#   --require-battery Fail validate if battery deltas missing
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
SCAN_SUBDIR="scan_a_all_models"
DEST="${REPO_ROOT}/Shared_pipeline_Files/results/device/${SCAN_SUBDIR}"

SKIP_STAGE=0
SKIP_BUILD=0
SKIP_PUSH=0
PULL_ONLY=0
MIN_SCANS=1
REQUIRE_BATTERY=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --skip-stage) SKIP_STAGE=1 ;;
    --skip-build) SKIP_BUILD=1 ;;
    --skip-push) SKIP_PUSH=1 ;;
    --pull-only) PULL_ONLY=1 ;;
    --min-scans) MIN_SCANS="${2:?}"; shift ;;
    --require-battery) REQUIRE_BATTERY=1 ;;
    -h|--help)
      sed -n '2,20p' "$0"
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
  if [[ "$SKIP_STAGE" -eq 0 ]]; then
    echo "=== Staging all model bundles ==="
    "$REPO_ROOT/Shared_pipeline_Files/tools/stage_all_models.sh"
  fi

  if [[ "$SKIP_BUILD" -eq 0 ]]; then
    echo "=== Building and installing debug APK ==="
    (cd "$REPO_ROOT/vigidroid" && ./gradlew installDebug)
  fi

  if [[ "$SKIP_PUSH" -eq 0 ]]; then
    if adb get-state >/dev/null 2>&1; then
      echo "=== Pushing device eval APKs to Download/Scanable ==="
      "$REPO_ROOT/Android_Works/push_device_eval_apks.sh"
    else
      echo "No adb device — skip push (APKs must already be on phone)."
    fi
  fi

  cat <<EOF

=== Manual step (POCO F3) ===
1. Open VigiDroid → set Scan mode to "Ablation (all models)" (cascade OFF).
2. Tap "Rescan all" (scans Download/*.apk and Download/Scanable/*.apk).
3. Wait until the scan queue finishes.

Then re-run with --pull-only:
  $0 --pull-only --min-scans 400

EOF
fi

if ! adb get-state >/dev/null 2>&1; then
  echo "adb device not connected — cannot pull metrics." >&2
  exit 1
fi

echo "=== Pulling device metrics → ${SCAN_SUBDIR}/ ==="
"$SCRIPT_DIR/pull_device_metrics.sh" "$SCAN_SUBDIR"

METRICS=""
if [[ -f "$DEST/scan_a_all_models.jsonl" ]]; then
  METRICS="$DEST/scan_a_all_models.jsonl"
elif [[ -f "$DEST/scan_a_all_models.json" ]]; then
  METRICS="$DEST/scan_a_all_models.json"
elif [[ -f "$DEST/all_scan_metrics.jsonl" ]]; then
  METRICS="$DEST/all_scan_metrics.jsonl"
elif [[ -f "$DEST/all_scan_metrics.json" ]]; then
  METRICS="$DEST/all_scan_metrics.json"
else
  echo "No metrics file under $DEST" >&2
  exit 1
fi

VALIDATE_ARGS=(--min-scans "$MIN_SCANS" --write-table "$REPO_ROOT/Shared_pipeline_Files/results/figures/plot_metrics_table.json")
if [[ "$REQUIRE_BATTERY" -eq 1 ]]; then
  VALIDATE_ARGS+=(--require-battery)
fi

echo "=== Validating Scan A ==="
python3 "$SCRIPT_DIR/validate_scan_a.py" "$METRICS" "${VALIDATE_ARGS[@]}"
