#!/usr/bin/env bash
# A4 CI gate — mldp_dexheader_cascade device parity (extract + ONNX ±1e-4).
# Requires: connected device/emulator, Android SDK, Gradle wrapper in vigidroid/.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VIGIDROID="$REPO_ROOT/vigidroid"
TEST_CLASS="com.msh.vigidroid.MldpDexHeaderA4ParityTest"

echo "=== MLDP + Dex header cascade A4 parity gate ==="
echo "Test class: $TEST_CLASS"
echo ""

if [[ ! -f "$VIGIDROID/gradlew" ]]; then
  echo "ERROR: vigidroid/gradlew not found"
  exit 1
fi

echo "Refreshing A4 parity fixtures (A1 APKs + A2 vectors + expected scores)..."
bash "$REPO_ROOT/mldp_dexheader_cascade/scripts/generate_a4_parity_fixtures.sh"

cd "$VIGIDROID"

echo ""
echo "Installing debug APK + androidTest APK..."
./gradlew :app:installDebug :app:installDebugAndroidTest

echo ""
echo "Running instrumented parity tests on device..."
./gradlew :app:connectedDebugAndroidTest \
  -Pandroid.testInstrumentationRunnerArguments.class="$TEST_CLASS"

echo ""
echo "A4 PASSED — mldp_dexheader_cascade device parity within 1e-4"
