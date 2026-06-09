#!/usr/bin/env bash
# Unified ONNX export driver — wired in Phase 4 after D3–D5 training completes.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
ASSETS="${REPO_ROOT}/vigidroid/app/src/main/assets/models"

echo "=== export_all_onnx.sh (skeleton) ==="
echo "Repo: $REPO_ROOT"
echo "Target assets: $ASSETS"
echo

run_export() {
  local name="$1"
  local cmd="$2"
  if [[ -f "${cmd%% *}" || -x "$(dirname "$cmd")" ]]; then
    echo "--- $name ---"
    (cd "$REPO_ROOT" && eval "$cmd") || echo "WARN: $name export failed (checkpoint missing?)"
  else
    echo "SKIP $name: script not found"
  fi
}

mkdir -p "$ASSETS"

# D2 — already deployed at assets root; optional re-export
run_export "D2 ByteCNN" "python 1dcnn/export_onnx.py"

# D3–D5 — scripts added in Phase 4
run_export "D3 MLP(H)" "python Dex_header_paper_implementation/only_base1_model/scripts/export_onnx.py"
run_export "D4 Early-Fusion Dex+Manifest" "python Dex_header_paper_implementation/custom_approach/full_combined_pipeline_approach/scripts/export_onnx.py"
run_export "D5 Dual-Branch Dex+Manifest" "python Dex_header_paper_implementation/custom_approach/dual_branch_merge_approach/scripts/export_onnx.py"

echo
echo "Copy export bundles to $ASSETS/<model_id>/ when Phase 4 scripts exist."
echo "Run parity: Shared_pipeline_Files/tools/run_parity_all.sh"
