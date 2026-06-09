#!/usr/bin/env bash
# Stage mlp_header ONNX export bundle into vigidroid assets (P7 → A1–A4).
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# shellcheck source=/dev/null
source "$REPO_ROOT/Shared_pipeline_Files/tools/stage_common.sh"
stage_common_init

SRC="$REPO_ROOT/Dex_header_paper_implementation/only_base1_model/artifacts/export/mlp_header"
stage_standard_onnx_bundle "mlp_header" "$SRC"
echo "Done."
