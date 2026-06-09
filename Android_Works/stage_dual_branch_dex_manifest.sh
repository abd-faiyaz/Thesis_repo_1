#!/usr/bin/env bash
# Stage dual_branch_dex_manifest ONNX export bundle into vigidroid assets (P7 → A1–A4).
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# shellcheck source=/dev/null
source "$REPO_ROOT/Shared_pipeline_Files/tools/stage_common.sh"
stage_common_init

SRC="$REPO_ROOT/Dex_header_paper_implementation/custom_approach/dual_branch_merge_approach/artifacts/export/dual_branch_dex_manifest"
stage_standard_onnx_bundle "dual_branch_dex_manifest" "$SRC"
echo "Done."
