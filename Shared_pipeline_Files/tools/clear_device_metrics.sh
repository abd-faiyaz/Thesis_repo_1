#!/usr/bin/env bash
# Remove on-device scan metrics (split by scan mode on Phase 2+ APK).
#
# Usage:
#   ./clear_device_metrics.sh           # both Scan A + B + legacy files
#   ./clear_device_metrics.sh ablation  # Scan A only (scan_a_all_models.jsonl)
#   ./clear_device_metrics.sh cascade   # Scan B only (scan_b_cascade.jsonl)
set -euo pipefail

PKG="com.msh.vigidroid"
MODE="${1:-all}"
DIRS=(
  "/sdcard/Android/data/${PKG}/files/metrics"
  "/storage/emulated/0/Android/data/${PKG}/files/metrics"
)

SCAN_A_FILES=(
  "scan_a_all_models.jsonl"
  "scan_a_all_models.json"
)
SCAN_B_FILES=(
  "scan_b_cascade.jsonl"
  "scan_b_cascade.json"
)
LEGACY_FILES=(
  "all_scan_metrics.jsonl"
  "all_scan_metrics.json"
)

if ! adb get-state >/dev/null 2>&1; then
  echo "adb device not connected" >&2
  exit 1
fi

case "$MODE" in
  ablation|scan_a|a) FILES=("${SCAN_A_FILES[@]}") ;;
  cascade|scan_b|b) FILES=("${SCAN_B_FILES[@]}") ;;
  all) FILES=("${SCAN_A_FILES[@]}" "${SCAN_B_FILES[@]}" "${LEGACY_FILES[@]}") ;;
  *)
    echo "Usage: $0 [all|ablation|cascade]" >&2
    exit 2
    ;;
esac

CLEARED=0
for DIR in "${DIRS[@]}"; do
  if adb shell "test -d '$DIR'" 2>/dev/null; then
    echo "Clearing in $DIR (mode=$MODE)"
    for f in "${FILES[@]}"; do
      adb shell "rm -f '$DIR/$f'" 2>/dev/null || true
    done
    CLEARED=1
  fi
done

if [[ "$CLEARED" -eq 0 ]]; then
  echo "No metrics directory found on device."
else
  echo "Cleared device metrics ($MODE)."
fi
