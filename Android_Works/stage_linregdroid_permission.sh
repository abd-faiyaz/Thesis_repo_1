#!/usr/bin/env bash
# Stage linregdroid_permission ONNX export bundle into vigidroid assets (P7 → A1–A4).
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# shellcheck source=/dev/null
source "$REPO_ROOT/Shared_pipeline_Files/tools/stage_common.sh"
stage_common_init

SRC="$REPO_ROOT/linear/artifacts/export/linregdroid_permission"
stage_standard_onnx_bundle "linregdroid_permission" "$SRC"
echo "Done."
