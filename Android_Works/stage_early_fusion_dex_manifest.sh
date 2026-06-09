#!/usr/bin/env bash
# Stage early_fusion_dex_manifest ONNX export bundle into vigidroid assets (P7 → A1–A4).
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# shellcheck source=/dev/null
source "$REPO_ROOT/Shared_pipeline_Files/tools/stage_common.sh"
stage_common_init

SRC="$REPO_ROOT/Dex_header_paper_implementation/custom_approach/full_combined_pipeline_approach/artifacts/export/early_fusion_dex_manifest"
stage_standard_onnx_bundle "early_fusion_dex_manifest" "$SRC"
echo "Done."
