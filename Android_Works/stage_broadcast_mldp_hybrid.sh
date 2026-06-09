#!/usr/bin/env bash
# Stage broadcast_mldp_hybrid ONNX export bundle into vigidroid assets (P7 → A1–A4).
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# shellcheck source=/dev/null
source "$REPO_ROOT/Shared_pipeline_Files/tools/stage_common.sh"
stage_common_init

SRC="$REPO_ROOT/broadcast_mldp_hybrid/artifacts/export/broadcast_mldp_hybrid"
stage_standard_onnx_bundle "broadcast_mldp_hybrid" "$SRC"

echo "Regenerating A4 androidTest manifests (requires APK corpus on disk)..."
bash "$REPO_ROOT/broadcast_mldp_hybrid/scripts/generate_a4_parity_fixtures.sh" || {
  echo "WARN: A4 fixture generation skipped (APK corpus unavailable). Existing androidTest assets kept."
}

echo "Done."
