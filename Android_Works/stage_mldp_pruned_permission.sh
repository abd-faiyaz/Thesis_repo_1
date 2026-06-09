#!/usr/bin/env bash
# Stage mldp_pruned_permission ONNX export bundle into vigidroid assets (P7 → A1–A4).
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# shellcheck source=/dev/null
source "$REPO_ROOT/Shared_pipeline_Files/tools/stage_common.sh"
stage_common_init

SRC="$REPO_ROOT/permission_extractor/artifacts/export/mldp_pruned_permission"
stage_standard_onnx_bundle "mldp_pruned_permission" "$SRC"
echo "Done."
