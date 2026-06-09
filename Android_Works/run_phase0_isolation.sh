#!/usr/bin/env bash
# Phase 0 — isolate ONNX infer vs extract for the three failing models.
# Requires: connected device/emulator, Android SDK, Gradle wrapper in vigidroid/.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VIGIDROID="$REPO_ROOT/vigidroid"
TEST_CLASS="com.msh.vigidroid.Phase0FailingModelsIsolationTest"

echo "=== Phase 0: failing-model ONNX isolation ==="
echo "Test class: $TEST_CLASS"
echo ""
echo "If these tests fail with f != java.lang.Long, the bug is in ONNX infer (not APK extract)."
echo "If they pass, re-scan an APK and check scan detail for @extract vs @infer in error text."
echo ""

if [[ ! -f "$VIGIDROID/gradlew" ]]; then
  echo "ERROR: vigidroid/gradlew not found"
  exit 1
fi

if ! command -v adb >/dev/null 2>&1; then
  echo "WARNING: adb not found — install Android platform-tools"
elif ! adb devices | awk 'NR>1 && $2=="device"{found=1} END{exit !found}'; then
  echo "ERROR: no adb device connected (adb devices)"
  exit 1
fi

cd "$VIGIDROID"

echo "Installing debug APK + androidTest APK..."
./gradlew :app:installDebug :app:installDebugAndroidTest

echo ""
echo "Running Phase 0 isolation tests..."
./gradlew :app:connectedDebugAndroidTest \
  -Pandroid.testInstrumentationRunnerArguments.class="$TEST_CLASS"

echo ""
echo "Optional: full A4 gates for the same models"
echo "  $REPO_ROOT/Android_Works/run_broadcast_mldp_hybrid_a4.sh"
echo "  $REPO_ROOT/Android_Works/run_mldp_dexheader_a4.sh"
echo ""
echo "Phase 0 isolation tests finished — check logcat tag StageDiagnostics / Phase0Isolation"
