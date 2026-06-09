#!/usr/bin/env bash
# =============================================================================
# ultimate_runner.sh
#
# One-shot orchestrator for a fresh PC: train all thesis models (P0–P8),
# stage ONNX bundles into vigidroid/, build cascade_policy.json, and leave the
# Android app ready to build/install on a phone.
#
# Prerequisites:
#   - Linux, bash, python3, git
#   - Full APK corpus on disk (benign/ + malware/ year folders)
#   - NVIDIA GPU recommended (CPU works but very slow)
#
# Usage:
#   export APK_ROOT=/path/to/android-apks
#   ./ultimate_runner.sh
#
# Optional:
#   STAGE_ANDROID=0          — PC-only; skip copying bundles to vigidroid/
#   SKIP_SETUP=1             — skip thesis_venv creation if already done
#   SKIP_CALIBRATION=1       — skip collect_calibration + cascade_policy.json
#   SKIP_ARCHIVE=1           — disable per-model run archives + thesis figures
#   SKIP_PLOTS=1             — keep archives but skip PNG figure generation
#   ULTIMATE_LOG=path.log    — tee full output to this file
#
# Wall-clock: days on full corpus if run sequentially.
# =============================================================================

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_ROOT"

APK_ROOT="${APK_ROOT:-}"
STAGE_ANDROID="${STAGE_ANDROID:-1}"
SKIP_SETUP="${SKIP_SETUP:-0}"
SKIP_CALIBRATION="${SKIP_CALIBRATION:-0}"
SKIP_ARCHIVE="${SKIP_ARCHIVE:-0}"
SKIP_PLOTS="${SKIP_PLOTS:-0}"
ULTIMATE_LOG="${ULTIMATE_LOG:-}"

export APK_ROOT STAGE_ANDROID SKIP_ARCHIVE SKIP_PLOTS

# Propagate a clean full-run environment (no smoke/quick/limit skips).
unset PREPROCESS_LIMIT QUICK SMOKE EXTRACT_LIMIT DEX_STATS_LIMIT || true

section() {
  echo ""
  echo "============================================================================="
  echo "  $1"
  echo "============================================================================="
}

if [[ -n "$ULTIMATE_LOG" ]]; then
  mkdir -p "$(dirname "$ULTIMATE_LOG")"
  exec > >(tee -a "$ULTIMATE_LOG") 2>&1
fi

section "Ultimate runner — configuration"
echo "REPO_ROOT:         $REPO_ROOT"
echo "APK_ROOT:          ${APK_ROOT:-<not set>}"
echo "STAGE_ANDROID:     $STAGE_ANDROID"
echo "SKIP_SETUP:        $SKIP_SETUP"
echo "SKIP_CALIBRATION:  $SKIP_CALIBRATION"
echo "SKIP_ARCHIVE:      $SKIP_ARCHIVE"
echo "SKIP_PLOTS:        $SKIP_PLOTS"
echo "ULTIMATE_LOG:      ${ULTIMATE_LOG:-<none>}"

if [[ -z "$APK_ROOT" ]]; then
  echo "ERROR: APK_ROOT is required." >&2
  echo "  export APK_ROOT=/path/to/android-apks" >&2
  exit 1
fi
if [[ ! -d "$APK_ROOT" ]]; then
  echo "ERROR: APK_ROOT does not exist: $APK_ROOT" >&2
  exit 1
fi

section "Step 0 — Environment setup"
if [[ "$SKIP_SETUP" != "1" ]]; then
  bash "$REPO_ROOT/scripts/setup_thesis_venv.sh"
else
  echo "(Skipping setup; SKIP_SETUP=1)"
fi

export PYTHON="${PYTHON:-$REPO_ROOT/thesis_venv/bin/python}"
if [[ ! -x "$PYTHON" ]]; then
  echo "ERROR: thesis_venv python not found at $PYTHON" >&2
  exit 1
fi

"$PYTHON" -c "import torch, sklearn, onnxruntime; import shared_calibration, shared_splits; print('Env OK — torch', torch.__version__)"

chmod +x \
  "$REPO_ROOT/permission_extractor/run_mldp_pruned_perm_cl.sh" \
  "$REPO_ROOT/linear/run_linregdroid.sh" \
  "$REPO_ROOT/broadcast_mldp_hybrid/run_brd_mldp_hybrid.sh" \
  "$REPO_ROOT/mldp_dexheader_cascade/run_mldp_dexheader_cascade.sh" \
  "$REPO_ROOT/Dex_header_paper_implementation/only_base1_model/run_base_model_1.sh" \
  "$REPO_ROOT/Dex_header_paper_implementation/custom_approach/full_combined_pipeline_approach/run_pattern_a.sh" \
  "$REPO_ROOT/Dex_header_paper_implementation/custom_approach/dual_branch_merge_approach/run_pattern_b.sh" \
  "$REPO_ROOT/dexheader_broadcast_fusion/run_dexheader_broadcast_fusion.sh" \
  "$REPO_ROOT/Android_Works"/stage_*.sh \
  "$REPO_ROOT/Shared_pipeline_Files/tools/stage_all_models.sh" \
  2>/dev/null || true

run_pipeline() {
  local name="$1"
  local dir="$2"
  local script="$3"
  section "$name"
  cd "$REPO_ROOT/$dir"
  export APK_ROOT STAGE_ANDROID
  bash "./$script"
  cd "$REPO_ROOT"
}

section "Step 1 — permission_extractor → mldp_pruned_permission"
run_pipeline "Pipeline 1/8" "permission_extractor" "run_mldp_pruned_perm_cl.sh"

section "Step 2 — Canonical val manifest (cross-model calibration alignment)"
"$PYTHON" "$REPO_ROOT/Shared_pipeline_Files/tools/build_canonical_val_manifest.py" \
  --source-manifest "$REPO_ROOT/permission_extractor/artifacts/processed/manifest_val.json"

run_pipeline "Pipeline 2/8 — linregdroid_permission" "linear" "run_linregdroid.sh"
run_pipeline "Pipeline 3/8 — broadcast_mldp_hybrid" "broadcast_mldp_hybrid" "run_brd_mldp_hybrid.sh"
run_pipeline "Pipeline 4/8 — mlp_header" \
  "Dex_header_paper_implementation/only_base1_model" "run_base_model_1.sh"
run_pipeline "Pipeline 5/8 — mldp_dexheader_cascade" "mldp_dexheader_cascade" "run_mldp_dexheader_cascade.sh"
run_pipeline "Pipeline 6/8 — early_fusion_dex_manifest" \
  "Dex_header_paper_implementation/custom_approach/full_combined_pipeline_approach" "run_pattern_a.sh"
run_pipeline "Pipeline 7/8 — dual_branch_dex_manifest" \
  "Dex_header_paper_implementation/custom_approach/dual_branch_merge_approach" "run_pattern_b.sh"
run_pipeline "Pipeline 8/8 — dexheader_broadcast_fusion" \
  "dexheader_broadcast_fusion" "run_dexheader_broadcast_fusion.sh"

section "Step 3 — Refresh all Android model assets"
if [[ "$STAGE_ANDROID" != "0" ]]; then
  bash "$REPO_ROOT/Shared_pipeline_Files/tools/stage_all_models.sh"
else
  echo "(Skipping batch staging; STAGE_ANDROID=0)"
fi

if [[ "$SKIP_CALIBRATION" != "1" ]]; then
  section "Step 4 — Cross-model cascade policy"
  "$PYTHON" "$REPO_ROOT/Shared_pipeline_Files/tools/collect_calibration_val_scores.py"
  "$PYTHON" "$REPO_ROOT/Shared_pipeline_Files/tools/build_cascade_policy.py" \
    --workspace "$REPO_ROOT/Shared_pipeline_Files/calibration" \
    --tier-spec "$REPO_ROOT/Shared_pipeline_Files/data/cascade_tier_spec.json" \
    --out "$REPO_ROOT/Shared_pipeline_Files/calibration/cascade_policy.json"
  if [[ "$STAGE_ANDROID" != "0" ]]; then
    cp "$REPO_ROOT/Shared_pipeline_Files/calibration/cascade_policy.json" \
      "$REPO_ROOT/vigidroid/app/src/main/assets/cascade_policy.json"
    echo "Copied cascade_policy.json → vigidroid/app/src/main/assets/"
  fi
else
  echo "(Skipping calibration; SKIP_CALIBRATION=1)"
fi

section "Step 5 — Asset verification"
ASSETS="$REPO_ROOT/vigidroid/app/src/main/assets/models"
MODEL_IDS=(
  mldp_pruned_permission
  linregdroid_permission
  broadcast_mldp_hybrid
  mlp_header
  mldp_dexheader_cascade
  early_fusion_dex_manifest
  dual_branch_dex_manifest
  dexheader_broadcast_fusion
)

missing=0
for model_id in "${MODEL_IDS[@]}"; do
  if [[ "$model_id" == "mldp_dexheader_cascade" ]]; then
    if [[ -f "$ASSETS/$model_id/mode_a/model.onnx" ]]; then
      echo "  OK  $model_id"
    else
      echo "  MISSING  $model_id"
      missing=$((missing + 1))
    fi
  elif [[ -f "$ASSETS/$model_id/model.onnx" ]]; then
    echo "  OK  $model_id"
  else
    echo "  MISSING  $model_id"
    missing=$((missing + 1))
  fi
done

section "Ultimate runner finished"
echo "VigiDroid project: $REPO_ROOT/vigidroid/"
echo "Model assets:      $ASSETS/"
if [[ "$missing" -gt 0 ]]; then
  echo "WARN: $missing model bundle(s) missing under assets (see verification above)."
  exit 1
fi
echo ""
echo "Next steps:"
echo "  1. Open vigidroid/ in Android Studio"
echo "  2. Build & install on your phone"
echo "  3. Run A4 instrumented parity tests (Android_Works/run_*_a4.sh)"
echo "  4. Scan APKs from the app"
echo ""
echo "Done."
