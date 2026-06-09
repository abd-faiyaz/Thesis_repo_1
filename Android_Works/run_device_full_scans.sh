#!/usr/bin/env bash
# Trigger Scan A (ablation) and Scan B (cascade) on a connected phone via adb.
#
# Prerequisites: eval APKs on device (Download/ + Download/Scanable/), debug APK installed.
#
# Usage:
#   ./run_device_full_scans.sh              # both scans sequentially
#   ./run_device_full_scans.sh --scan-a-only
#   ./run_device_full_scans.sh --scan-b-only
#   ./run_device_full_scans.sh --expected-apks 1555
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PKG="com.msh.vigidroid"
SERVICE="${PKG}/.ScanService"
METRICS_DIRS=(
  "/sdcard/Android/data/${PKG}/files/metrics"
  "/storage/emulated/0/Android/data/${PKG}/files/metrics"
)

SCAN_A_ONLY=0
SCAN_B_ONLY=0
EXPECTED_APKS=1555
POLL_SECS=20

while [[ $# -gt 0 ]]; do
  case "$1" in
    --scan-a-only) SCAN_A_ONLY=1 ;;
    --scan-b-only) SCAN_B_ONLY=1 ;;
    --expected-apks) EXPECTED_APKS="${2:?}"; shift ;;
    -h|--help)
      sed -n '2,12p' "$0"
      exit 0
      ;;
    *)
      echo "Unknown option: $1" >&2
      exit 2
      ;;
  esac
  shift
done

if [[ "$SCAN_A_ONLY" -eq 1 && "$SCAN_B_ONLY" -eq 1 ]]; then
  echo "Choose at most one of --scan-a-only / --scan-b-only" >&2
  exit 2
fi

if ! adb get-state >/dev/null 2>&1; then
  echo "adb device not connected" >&2
  exit 1
fi

metrics_file_on_device() {
  local name="$1"
  for dir in "${METRICS_DIRS[@]}"; do
    if adb shell "test -f '$dir/$name'" 2>/dev/null; then
      echo "$dir/$name"
      return 0
    fi
  done
  return 1
}

count_jsonl_lines() {
  local remote="$1"
  adb shell "wc -l < '$remote'" 2>/dev/null | tr -d ' \r' || echo 0
}

wait_for_scan() {
  local label="$1"
  local jsonl_name="$2"
  local remote=""
  local last=0
  local stable=0
  local deadline=$((SECONDS + 7200))

  echo "=== Waiting for $label ($jsonl_name, target ~$EXPECTED_APKS scan lines) ==="
  adb logcat -c 2>/dev/null || true

  while [[ $SECONDS -lt $deadline ]]; do
    if remote="$(metrics_file_on_device "$jsonl_name")"; then
      local lines
      lines="$(count_jsonl_lines "$remote")"
      echo "$(date -Is)  $label lines=$lines  ($remote)"
      if [[ "$lines" -ge "$EXPECTED_APKS" ]]; then
        echo "$label reached target line count."
        return 0
      fi
      if [[ "$lines" == "$last" && "$lines" -gt 0 ]]; then
        stable=$((stable + 1))
        if [[ $stable -ge 3 ]]; then
          echo "$label line count stable at $lines — checking logcat for session complete."
          if adb logcat -d -s ScanService:V 2>/dev/null | grep -q "Session complete"; then
            return 0
          fi
        fi
      else
        stable=0
      fi
      last="$lines"
    else
      echo "$(date -Is)  $label — metrics file not created yet"
    fi
    sleep "$POLL_SECS"
  done
  echo "TIMEOUT waiting for $label" >&2
  return 1
}

trigger_scan() {
  local cascade="$1"
  echo "=== Triggering scan via MainActivity adb intent (cascade=$cascade) ==="
  adb shell am start -n "${PKG}/.MainActivity" \
    --ez auto_rescan_all true \
    --ez cascade_enabled "$cascade" \
    --activity-clear-top
}

# Keep CPU awake during long batch (best effort).
adb shell svc power stayon usb 2>/dev/null || true

# Warm ONNX pipelines (ScanService.onCreate).
adb shell am start -n "${PKG}/.MainActivity" >/dev/null 2>&1 || true
sleep 5

RUN_A=1
RUN_B=1
[[ "$SCAN_B_ONLY" -eq 1 ]] && RUN_A=0
[[ "$SCAN_A_ONLY" -eq 1 ]] && RUN_B=0

if [[ "$RUN_A" -eq 1 ]]; then
  "$REPO_ROOT/Shared_pipeline_Files/tools/clear_device_metrics.sh" ablation
  trigger_scan false
  wait_for_scan "Scan A (ablation)" "scan_a_all_models.jsonl"
fi

if [[ "$RUN_B" -eq 1 ]]; then
  "$REPO_ROOT/Shared_pipeline_Files/tools/clear_device_metrics.sh" cascade
  trigger_scan true
  wait_for_scan "Scan B (cascade)" "scan_b_cascade.jsonl"
fi

adb shell svc power stayon false 2>/dev/null || true
echo "Device scans finished."
