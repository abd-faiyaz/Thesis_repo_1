#!/usr/bin/env bash
# Phase 2 — extraction hardening: asset config + FeatureContext parity on device.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VIGIDROID="$REPO_ROOT/vigidroid"
APK_SRC="/mnt/Files/thesis_full_dataset/2023/malware/E780ECFBB6C1F3F9DCCDB3AB7F0F88D4AF02FF2EAFB42AD1A6CBA4DA4B8C1D51.apk"
APK_NAME="scan_1514_malware.apk"

if ! command -v adb >/dev/null 2>&1 || ! adb devices | awk 'NR>1 && $2=="device"{found=1} END{exit !found}'; then
  echo "ERROR: no adb device connected"
  exit 1
fi

if [[ -f "$APK_SRC" ]]; then
  echo "=== Push eval APK for FeatureContext tests ==="
  adb push "$APK_SRC" "/data/local/tmp/$APK_NAME"
fi

cd "$VIGIDROID"
echo "=== Install debug + androidTest APKs ==="
./gradlew :app:installDebug :app:installDebugAndroidTest

TEST_CLASSES="com.msh.vigidroid.Phase2AssetConfigTest,com.msh.vigidroid.Phase2FeatureContextExtractionTest"

echo "=== Phase 2 extraction tests ==="
./gradlew :app:connectedDebugAndroidTest \
  -Pandroid.testInstrumentationRunnerArguments.class="$TEST_CLASSES"

echo ""
echo "Phase 2 extraction verification PASSED"
