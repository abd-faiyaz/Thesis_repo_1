#!/usr/bin/env bash
# Pull device scan JSON from phone/emulator to Shared_pipeline_Files/results/device/
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
DEST="${REPO_ROOT}/Shared_pipeline_Files/results/device"
PKG="com.msh.vigidroid"

mkdir -p "$DEST"

# Try common external-files paths (scoped storage)
CANDIDATES=(
  "/sdcard/Android/data/${PKG}/files/metrics"
  "/storage/emulated/0/Android/data/${PKG}/files/metrics"
)

PULLED=0
for SRC in "${CANDIDATES[@]}"; do
  if adb shell "test -d '$SRC'" 2>/dev/null; then
    echo "Pulling from $SRC → $DEST"
    adb pull "$SRC/." "$DEST/"
    PULLED=1
    break
  fi
done

if [[ "$PULLED" -eq 0 ]]; then
  echo "Metrics dir not found on device. Scan an APK first, then retry." >&2
  echo "Expected one of:" >&2
  printf '  %s\n' "${CANDIDATES[@]}" >&2
  exit 1
fi

echo "Done. Files in $DEST"
ls -la "$DEST"
