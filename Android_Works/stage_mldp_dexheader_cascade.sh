#!/usr/bin/env bash
# Stage mldp_dexheader_cascade ONNX export bundle into vigidroid assets (P7 → A1–A4).
# PC pipeline staging only — does not run Gradle/JVM tests (use run_mldp_dexheader_a2.sh for that).
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

echo "Staging mldp_dexheader_cascade"
bash "$REPO_ROOT/mldp_dexheader_cascade/scripts/deploy_android_assets.sh"

echo "Regenerating A1 JVM + androidTest extraction fixtures (requires APK corpus on disk)..."
bash "$REPO_ROOT/mldp_dexheader_cascade/scripts/generate_a1_parity_fixtures.sh" || {
  echo "WARN: A1 fixture generation skipped (APK corpus unavailable or val paths missing)."
  echo "      Run manually after corpus is available:"
  echo "        bash mldp_dexheader_cascade/scripts/generate_a1_parity_fixtures.sh"
}

echo "Regenerating A4 androidTest fixtures (requires APK corpus on disk)..."
bash "$REPO_ROOT/mldp_dexheader_cascade/scripts/generate_a4_parity_fixtures.sh" || {
  echo "WARN: A4 fixture generation skipped (APK corpus unavailable). Existing androidTest assets kept."
}

echo "Done."
echo "Optional JVM smoke tests: bash Android_Works/run_mldp_dexheader_a2.sh"
echo "On-device A4 parity:       bash Android_Works/run_mldp_dexheader_a4.sh"
