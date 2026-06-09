#!/usr/bin/env bash
# P1 exit gate: push scan_1514_malware.apk and run full legacy scan on device.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VIGIDROID="$REPO_ROOT/vigidroid"
APK_SRC="/mnt/Files/thesis_full_dataset/2023/malware/E780ECFBB6C1F3F9DCCDB3AB7F0F88D4AF02FF2EAFB42AD1A6CBA4DA4B8C1D51.apk"
APK_NAME="scan_1514_malware.apk"
DEVICE_TMP="/data/local/tmp"
DEVICE_EVAL_DIR="/sdcard/Download/Scanable"

if [[ ! -f "$APK_SRC" ]]; then
  echo "ERROR: eval APK not found: $APK_SRC"
  exit 1
fi

if ! command -v adb >/dev/null 2>&1 || ! adb devices | awk 'NR>1 && $2=="device"{found=1} END{exit !found}'; then
  echo "ERROR: no adb device connected"
  exit 1
fi

echo "=== Push $APK_NAME to device (/data/local/tmp + Scanable) ==="
adb push "$APK_SRC" "$DEVICE_TMP/$APK_NAME"
adb shell "mkdir -p '$DEVICE_EVAL_DIR'"
adb push "$APK_SRC" "$DEVICE_EVAL_DIR/$APK_NAME"

cd "$VIGIDROID"
echo "=== Install debug + androidTest APKs ==="
./gradlew :app:installDebug :app:installDebugAndroidTest

echo "=== P1 exit: legacy all-models scan on real APK ==="
./gradlew :app:connectedDebugAndroidTest \
  -Pandroid.testInstrumentationRunnerArguments.class="com.msh.vigidroid.P1ExitLegacyScanTest"

echo ""
echo "P1 exit scan PASSED (11/11 stages ok on scan_1514_malware.apk)"
