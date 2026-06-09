#!/usr/bin/env bash
# Pull device scan JSON from phone/emulator to Shared_pipeline_Files/results/device/
#
# Usage:
#   ./pull_device_metrics.sh                         # whole metrics/ dir (legacy)
#   ./pull_device_metrics.sh scan_a_all_models       # Scan A file only
#   ./pull_device_metrics.sh scan_b_cascade           # Scan B file only
#   ./pull_device_metrics.sh scan_a_all_models ablation  # legacy: filter combined JSONL
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
SUBDIR="${1:-}"
MODE_FILTER="${2:-}"
if [[ -n "$SUBDIR" && "$SUBDIR" == /* ]]; then
  DEST="$SUBDIR"
else
  DEST="${REPO_ROOT}/Shared_pipeline_Files/results/device"
  if [[ -n "$SUBDIR" ]]; then
    DEST="${DEST}/${SUBDIR}"
  fi
fi
PKG="com.msh.vigidroid"

SCAN_A_JSONL="scan_a_all_models.jsonl"
SCAN_B_JSONL="scan_b_cascade.jsonl"
LEGACY_JSONL="all_scan_metrics.jsonl"

REMOTE_JSONL=""
case "${SUBDIR##*/}" in
  scan_a_all_models) REMOTE_JSONL="$SCAN_A_JSONL" ;;
  scan_b_cascade) REMOTE_JSONL="$SCAN_B_JSONL" ;;
esac

CANDIDATES=(
  "/sdcard/Android/data/${PKG}/files/metrics"
  "/storage/emulated/0/Android/data/${PKG}/files/metrics"
)

mkdir -p "$DEST"

pull_remote_file() {
  local remote_name="$1"
  local pulled=0
  for SRC in "${CANDIDATES[@]}"; do
    if adb shell "test -f '$SRC/$remote_name'" 2>/dev/null; then
      echo "Pulling $SRC/$remote_name → $DEST/"
      adb pull "$SRC/$remote_name" "$DEST/"
      pulled=1
      break
    fi
  done
  return "$((1 - pulled))"
}

PULLED=0
if [[ -n "$REMOTE_JSONL" ]]; then
  if pull_remote_file "$REMOTE_JSONL"; then
    PULLED=1
  elif [[ "$REMOTE_JSONL" == "$SCAN_A_JSONL" ]] && pull_remote_file "$LEGACY_JSONL"; then
    echo "Note: using legacy $LEGACY_JSONL (upgrade app for split files)."
    MODE_FILTER="${MODE_FILTER:-ablation}"
    PULLED=1
  elif [[ "$REMOTE_JSONL" == "$SCAN_B_JSONL" ]] && pull_remote_file "$LEGACY_JSONL"; then
    echo "Note: using legacy $LEGACY_JSONL (upgrade app for split files)."
    MODE_FILTER="${MODE_FILTER:-cascade}"
    PULLED=1
  fi
else
  for SRC in "${CANDIDATES[@]}"; do
    if adb shell "test -d '$SRC'" 2>/dev/null; then
      echo "Pulling from $SRC → $DEST"
      adb pull "$SRC/." "$DEST/"
      PULLED=1
      break
    fi
  done
fi

if [[ "$PULLED" -eq 0 ]]; then
  echo "Metrics not found on device. Scan an APK first, then retry." >&2
  exit 1
fi

echo "Done. Files in $DEST"
ls -la "$DEST"

if [[ -n "$REMOTE_JSONL" ]]; then
  JSONL="$DEST/$REMOTE_JSONL"
elif [[ -f "$DEST/$SCAN_A_JSONL" ]]; then
  JSONL="$DEST/$SCAN_A_JSONL"
elif [[ -f "$DEST/$LEGACY_JSONL" ]]; then
  JSONL="$DEST/$LEGACY_JSONL"
else
  JSONL="$DEST/$LEGACY_JSONL"
fi

if [[ -f "$JSONL" && -n "$MODE_FILTER" ]]; then
  echo "Filtering JSONL → ${MODE_FILTER} only"
  python3 "$SCRIPT_DIR/filter_device_pull.py" "$JSONL" --mode "$MODE_FILTER"
elif [[ -f "$JSONL" ]]; then
  OUT_JSON="${JSONL%.jsonl}.json"
  echo "Merging JSONL → $(basename "$OUT_JSON")"
  python3 "$SCRIPT_DIR/jsonl_to_json.py" "$JSONL" -o "$OUT_JSON"
else
  echo "Note: no JSONL merged (file missing or empty pull)."
fi
