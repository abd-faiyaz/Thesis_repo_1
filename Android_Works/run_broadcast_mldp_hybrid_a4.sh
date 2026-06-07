#!/usr/bin/env bash
# A4 CI gate — broadcast_mldp_hybrid device parity (extract + ONNX ±1e-4).
# Requires: connected device/emulator, Android SDK, Gradle wrapper in vigidroid/.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VIGIDROID="$REPO_ROOT/vigidroid"
TEST_CLASS="com.msh.vigidroid.BroadcastMldpHybridA4ParityTest"

echo "=== Broadcast + MLDP hybrid A4 parity gate ==="
echo "Test class: $TEST_CLASS"
echo ""

if [[ ! -f "$VIGIDROID/gradlew" ]]; then
  echo "ERROR: vigidroid/gradlew not found"
  exit 1
fi

cd "$VIGIDROID"

echo "Installing debug APK + androidTest APK..."
./gradlew :app:installDebug :app:installDebugAndroidTest

echo ""
echo "Running instrumented parity tests on device..."
./gradlew :app:connectedDebugAndroidTest \
  -Pandroid.testInstrumentationRunnerArguments.class="$TEST_CLASS"

echo ""
echo "A4 PASSED — broadcast_mldp_hybrid device parity within 1e-4"
