#!/usr/bin/env bash
# A2 — deploy ONNX assets, generate parity vectors, run JVM smoke tests.
set -euo pipefail
REPO="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO"

echo "=== Deploy export bundle to Android assets ==="
SRC="$REPO/mldp_dexheader_cascade/artifacts/export/mldp_dexheader_cascade"
DST="$REPO/vigidroid/app/src/main/assets/models/mldp_dexheader_cascade"
mkdir -p "$DST"
cp -r "$SRC/mode_a" "$SRC/mode_b" "$DST/"
cp "$SRC/thresholds.json" "$DST/"
cp -r "$SRC/features" "$DST/"
cp -r "$SRC/parity_samples" "$DST/"

echo "=== Generate parity_onnx_vectors.json ==="
bash "$REPO/mldp_dexheader_cascade/scripts/generate_a2_parity_vectors.sh" 2>/dev/null \
  || python3 "$REPO/mldp_dexheader_cascade/scripts/generate_a2_parity_vectors.py"

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
