#!/usr/bin/env bash
# Stage dexheader_broadcast_fusion ONNX export bundle into vigidroid assets (P7 → A1–A4).
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# shellcheck source=/dev/null
source "$REPO_ROOT/Shared_pipeline_Files/tools/stage_common.sh"
stage_common_init

SRC="$REPO_ROOT/dexheader_broadcast_fusion/artifacts/export/dexheader_broadcast_fusion"
stage_copytree_bundle "dexheader_broadcast_fusion" "$SRC"
echo "Done."
