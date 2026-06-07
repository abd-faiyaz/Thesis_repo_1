#!/usr/bin/env bash
# A1 — regenerate fixtures and run JVM + instrumented extraction parity.
set -euo pipefail
REPO="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO"

echo "=== Generate A1 parity fixtures (3 APKs + JSON) ==="
bash "$REPO/mldp_dexheader_cascade/scripts/generate_a1_parity_fixtures.sh"

echo "=== Copy golden manifests to JVM test resources ==="
FIXTURES="$REPO/vigidroid/app/src/androidTest/assets/models/mldp_dexheader_cascade/parity_samples"
JVM_MANIFESTS="$REPO/vigidroid/app/src/test/resources/mldp_dexheader_cascade/manifests"
mkdir -p "$JVM_MANIFESTS"
for i in 000 001 002; do
  cp "$FIXTURES/manifests/sample_${i}.xml" "$JVM_MANIFESTS/sample_${i}.xml"
done

echo "=== JVM unit tests ==="
cd "$REPO/vigidroid"
./gradlew :app:testDebugUnitTest --tests com.msh.vigidroid.MldpDexHeaderExtractorTest

echo "=== Instrumented A1 parity (device/emulator required) ==="
./gradlew :app:connectedDebugAndroidTest --tests com.msh.vigidroid.MldpDexHeaderA1ParityTest
