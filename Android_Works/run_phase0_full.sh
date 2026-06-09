#!/usr/bin/env bash
# Phase 0 complete verification: isolation tests + A4 parity gates for failing models.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VIGIDROID="$REPO_ROOT/vigidroid"

if [[ ! -f "$VIGIDROID/gradlew" ]]; then
  echo "ERROR: vigidroid/gradlew not found"
  exit 1
fi

if ! command -v adb >/dev/null 2>&1 || ! adb devices | awk 'NR>1 && $2=="device"{found=1} END{exit !found}'; then
  echo "ERROR: no adb device connected"
  exit 1
fi

cd "$VIGIDROID"

echo "=== Phase 0.3–0.6: isolation tests (ONNX-only + extract + e2e) ==="
./gradlew :app:installDebug :app:installDebugAndroidTest
./gradlew :app:connectedDebugAndroidTest \
  -Pandroid.testInstrumentationRunnerArguments.class="com.msh.vigidroid.Phase0FailingModelsIsolationTest"

echo ""
echo "=== Phase 0.1–0.2: broadcast_mldp_hybrid A4 ==="
./gradlew :app:connectedDebugAndroidTest \
  -Pandroid.testInstrumentationRunnerArguments.class="com.msh.vigidroid.BroadcastMldpHybridA4ParityTest"

echo ""
echo "=== Phase 0.1–0.2: mldp_dexheader Mode A/B A2 parity ==="
./gradlew :app:connectedDebugAndroidTest \
  -Pandroid.testInstrumentationRunnerArguments.class="com.msh.vigidroid.MldpDexHeaderA2ParityTest"

echo ""
echo "Phase 0 full verification PASSED"
echo "Next: install debug APK, re-scan scan_1514_malware.apk, check errors for @extract vs @infer tags."
