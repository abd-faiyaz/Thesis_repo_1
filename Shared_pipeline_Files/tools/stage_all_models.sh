#!/usr/bin/env bash
# Stage all thesis model ONNX export bundles into vigidroid app assets.
#
# Run once after P0–P8 end-to-end runners complete on the PC:
#   bash Shared_pipeline_Files/tools/stage_all_models.sh
#
# Optional:
#   STAGE_SKIP_MISSING=1  — warn and continue when a model export is missing
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
ANDROID_WORKS="$REPO_ROOT/Android_Works"
STAGE_SKIP_MISSING="${STAGE_SKIP_MISSING:-0}"

STAGE_SCRIPTS=(
  stage_mlp_header.sh
  stage_early_fusion_dex_manifest.sh
  stage_dual_branch_dex_manifest.sh
  stage_linregdroid_permission.sh
  stage_mldp_pruned_permission.sh
  stage_broadcast_mldp_hybrid.sh
  stage_mldp_dexheader_cascade.sh
  stage_dexheader_broadcast_fusion.sh
)

echo "=== stage_all_models.sh ==="
echo "Repo:   $REPO_ROOT"
echo "Target: $REPO_ROOT/vigidroid/app/src/main/assets/models/"
echo "STAGE_SKIP_MISSING: $STAGE_SKIP_MISSING"
echo

failed=0
for script in "${STAGE_SCRIPTS[@]}"; do
  path="$ANDROID_WORKS/$script"
  if [[ ! -x "$path" ]]; then
    echo "ERROR: missing or non-executable $path" >&2
  fi
  echo "--- $script ---"
  if bash "$path"; then
    echo
  elif [[ "$STAGE_SKIP_MISSING" == "1" ]]; then
    echo "WARN: $script failed — continuing (STAGE_SKIP_MISSING=1)"
    echo
    failed=$((failed + 1))
  else
    echo "ERROR: $script failed" >&2
    exit 1
  fi
done

echo "=== staging complete ==="
echo "Android assets: $REPO_ROOT/vigidroid/app/src/main/assets/models/"
if [[ "$failed" -gt 0 ]]; then
  echo "WARN: $failed model(s) skipped or failed (STAGE_SKIP_MISSING=1)"
fi
echo "Next: open vigidroid/ in Android Studio, build/install, run A4 parity tests, then scan."
