#!/usr/bin/env bash
# Stage LinRegDroid + MLDP-pruned permission bundles (legacy wrapper).
# Prefer individual scripts or Shared_pipeline_Files/tools/stage_all_models.sh.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
bash "$REPO_ROOT/Android_Works/stage_linregdroid_permission.sh"
bash "$REPO_ROOT/Android_Works/stage_mldp_pruned_permission.sh"
echo "Done. Android assets staged under vigidroid/app/src/main/assets/models/"
