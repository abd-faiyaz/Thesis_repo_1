#!/usr/bin/env bash
# A2 — deploy ONNX assets, refresh fixtures, run JVM smoke tests.
set -euo pipefail
REPO="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO"

SKIP_JVM_TESTS="${SKIP_JVM_TESTS:-0}"

bash "$REPO/mldp_dexheader_cascade/scripts/deploy_android_assets.sh"

echo "=== Regenerate A1 JVM + androidTest extraction fixtures ==="
bash "$REPO/mldp_dexheader_cascade/scripts/generate_a1_parity_fixtures.sh"

if [[ "$SKIP_JVM_TESTS" == "1" ]]; then
  echo "(Skipping JVM unit tests; SKIP_JVM_TESTS=1)"
  exit 0
fi

echo "=== JVM unit tests (JBR) ==="
export JAVA_HOME=/opt/android-studio/jbr
export PATH="$JAVA_HOME/bin:$PATH"
cd "$REPO/vigidroid"
./gradlew :app:testDebugUnitTest \
  --tests com.msh.vigidroid.MldpDexHeaderOnnxRunnerTest \
  --tests com.msh.vigidroid.MldpDexHeaderExtractorTest \
  --tests com.msh.vigidroid.ModelRegistryTest

echo ""
echo "On device/emulator:"
echo "  ./gradlew :app:connectedDebugAndroidTest --tests com.msh.vigidroid.MldpDexHeaderA2ParityTest"
