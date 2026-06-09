#!/usr/bin/env bash
# =============================================================================
# run_pattern_a.sh — End-to-end Pattern A (Full Combined Pipeline)
#
# Pipeline:
#   P0  verify_setup (+ optional pip install)
#   P2  preprocess APKs → per-APK shards (multi-dex sum-pool by default)
#   P5  train CombinedNet (resume-safe)
#   P6  evaluate best checkpoint (ACC, F1, AUC)
#   P7  export ONNX bundle (artifacts/export/early_fusion_dex_manifest/)
#   P8  PyTorch vs ONNX parity
#   +   figures, archive finalize, THESIS_SNIPPET (default on; PA_ARCHIVE=0 or SKIP_ARCHIVE=1 to disable)
#   +   Android asset staging by default (set STAGE_ANDROID=0 to skip)
#
# Usage:
#   ./run_pattern_a.sh
#   APK_ROOT=/data/apks ./run_pattern_a.sh
#   SKIP_PREPROCESS=1 ./run_pattern_a.sh
#   PREPROCESS_LIMIT=200 EPOCHS=2 ./run_pattern_a.sh
#   STAGE_ANDROID=0 ./run_pattern_a.sh
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$SCRIPT_DIR"
REPO_ROOT="$(cd "$ROOT/../../.." && pwd)"
cd "$ROOT"

# shellcheck source=/dev/null
source "$ROOT/scripts/activate_thesis_env.sh"

ARCHIVE_PY="$REPO_ROOT/scripts/thesis_run_archive.py"
PROFILE="early_fusion_dex_manifest"

APK_ROOT="${APK_ROOT:-$ROOT/data/apks}"
CONFIG="${CONFIG:-$ROOT/config/default.yaml}"
EPOCHS="${EPOCHS:-}"
INSTALL_DEPS="${INSTALL_DEPS:-0}"
VERIFY_SETUP="${VERIFY_SETUP:-1}"
SKIP_PREPROCESS="${SKIP_PREPROCESS:-0}"
SKIP_DEX_STATS="${SKIP_DEX_STATS:-0}"
SKIP_TRAIN="${SKIP_TRAIN:-0}"
SKIP_EVAL="${SKIP_EVAL:-0}"
SKIP_EXPORT_ONNX="${SKIP_EXPORT_ONNX:-0}"
SKIP_PARITY="${SKIP_PARITY:-0}"
SKIP_PACKAGE="${SKIP_PACKAGE:-0}"
SKIP_ARCHIVE="${SKIP_ARCHIVE:-0}"
SKIP_PLOTS="${SKIP_PLOTS:-0}"
FRESH_TRAIN="${FRESH_TRAIN:-0}"
PREPROCESS_LIMIT="${PREPROCESS_LIMIT:-}"
EXTRACT_LIMIT="${EXTRACT_LIMIT:-}"
DEX_STATS_LIMIT="${DEX_STATS_LIMIT:-}"
PIPELINE_LOG="${PIPELINE_LOG:-$ROOT/artifacts/pipeline.log}"

PA_ARCHIVE="${PA_ARCHIVE:-1}"
PA_RUN_ID="${PA_RUN_ID:-}"
STAGE_ANDROID="${STAGE_ANDROID:-1}"
if [[ "$SKIP_ARCHIVE" == "1" ]]; then
  PA_ARCHIVE=0
fi

export PYTHONPATH="${ROOT}${PYTHONPATH:+:$PYTHONPATH}"

section() {
  echo ""
  echo "============================================================================="
  echo "  $1"
  echo "============================================================================="
}

if [[ "$PA_ARCHIVE" == "1" ]]; then
  if [[ -z "$PA_RUN_ID" ]]; then
    PA_RUN_ID="run_$(date +%Y%m%d_%H%M%S)_pa"
  fi
  export PA_RUN_ID
  ARCHIVE_DIR="$ROOT/output_archives/$PA_RUN_ID"
  "$PYTHON" "$ARCHIVE_PY" bootstrap \
    --profile "$PROFILE" \
    --root "$ROOT" \
    --run-id "$PA_RUN_ID" \
    --config "$CONFIG" \
    ${APK_ROOT:+--apk-root "$APK_ROOT"}
  exec > >(tee -a "$ARCHIVE_DIR/logs/pipeline_full.log") 2>&1
else
  mkdir -p "$(dirname "$PIPELINE_LOG")"
  exec > >(tee -a "$PIPELINE_LOG") 2>&1
fi

section "Pattern A (Full Combined Pipeline) — configuration"
echo "ROOT:             $ROOT"
echo "APK_ROOT:         $APK_ROOT"
echo "PYTHON:           $PYTHON"
echo "THESIS_VENV:      ${THESIS_VENV:-<not set>}"
echo "CONFIG:           $CONFIG"
echo "SKIP_PREPROCESS:  $SKIP_PREPROCESS"
echo "SKIP_DEX_STATS:   $SKIP_DEX_STATS"
echo "SKIP_TRAIN:       $SKIP_TRAIN"
echo "SKIP_EVAL:        $SKIP_EVAL"
echo "SKIP_EXPORT:      $SKIP_EXPORT_ONNX"
echo "SKIP_PARITY:      $SKIP_PARITY"
echo "SKIP_PACKAGE:     $SKIP_PACKAGE"
echo "STAGE_ANDROID:    $STAGE_ANDROID"
echo "SKIP_ARCHIVE:     $SKIP_ARCHIVE"
echo "SKIP_PLOTS:       $SKIP_PLOTS"
echo "FRESH_TRAIN:      $FRESH_TRAIN"
echo "PA_ARCHIVE:       $PA_ARCHIVE"
echo "PA_RUN_ID:        ${PA_RUN_ID:-<auto when PA_ARCHIVE=1>}"
echo "PIPELINE_LOG:     $PIPELINE_LOG"

if [[ "$INSTALL_DEPS" == "1" ]]; then
  section "Install dependencies"
  _REQS="$(thesis_all_requirements_path)"
  echo "Using requirements: $_REQS"
  "$PYTHON" -m pip install -r "$_REQS"
fi

if [[ "$VERIFY_SETUP" == "1" ]]; then
  section "Verify environment"
  "$PYTHON" "$ROOT/scripts/verify_setup.py"
fi

if [[ "$SKIP_PREPROCESS" != "1" ]]; then
  section "Preprocess (scan → lexicon → norm → shards, multi-dex sum)"
  if [[ ! -d "$APK_ROOT" ]]; then
    echo "ERROR: APK_ROOT does not exist: $APK_ROOT"
    exit 1
  fi

  SCAN_ARGS=(--apk-root "$APK_ROOT")
  [[ -n "$CONFIG" ]] && SCAN_ARGS+=(--config "$CONFIG")
  [[ -n "$PREPROCESS_LIMIT" ]] && SCAN_ARGS+=(--limit "$PREPROCESS_LIMIT")
  "$PYTHON" -m src.preprocessing.scan_dataset "${SCAN_ARGS[@]}"

  LEX_ARGS=()
  [[ -n "$CONFIG" ]] && LEX_ARGS+=(--config "$CONFIG")
  "$PYTHON" -m src.preprocessing.build_lexicon "${LEX_ARGS[@]}"

  NORM_ARGS=()
  [[ -n "$CONFIG" ]] && NORM_ARGS+=(--config "$CONFIG")
  "$PYTHON" -m src.preprocessing.fit_header_norm "${NORM_ARGS[@]}"

  EXTRACT_ARGS=(--split all)
  [[ -n "$CONFIG" ]] && EXTRACT_ARGS+=(--config "$CONFIG")
  [[ -n "$EXTRACT_LIMIT" ]] && EXTRACT_ARGS+=(--limit "$EXTRACT_LIMIT")
  "$PYTHON" -m src.preprocessing.extract_to_cache "${EXTRACT_ARGS[@]}"
else
  echo "(Skipping preprocess; SKIP_PREPROCESS=1)"
fi

MANIFEST_TRAIN="$ROOT/artifacts/processed/manifest_train.json"
MANIFEST_VAL="$ROOT/artifacts/processed/manifest_val.json"
MANIFEST_TEST="$ROOT/artifacts/processed/manifest_test.json"
if [[ ! -f "$MANIFEST_TRAIN" ]] || [[ ! -f "$MANIFEST_VAL" ]] || [[ ! -f "$MANIFEST_TEST" ]]; then
  echo "ERROR: Missing manifests (train, val, test). Run preprocessing first."
  exit 1
fi

"$PYTHON" "$ARCHIVE_PY" export-corpus-stats \
  --profile "$PROFILE" \
  --root "$ROOT" \
  ${PA_RUN_ID:+--run-id "$PA_RUN_ID"}

if [[ "$SKIP_DEX_STATS" != "1" ]]; then
  section "Dex file count histogram (train split)"
  DEX_ARGS=(--split train)
  [[ -n "$CONFIG" ]] && DEX_ARGS+=(--config "$CONFIG")
  [[ -n "$DEX_STATS_LIMIT" ]] && DEX_ARGS+=(--limit "$DEX_STATS_LIMIT")
  "$PYTHON" "$ROOT/scripts/compute_dex_stats.py" "${DEX_ARGS[@]}" || true
else
  echo "(Skipping dex stats; SKIP_DEX_STATS=1)"
fi

section "Class balance (train split)"
BAL_ARGS=()
[[ -n "$CONFIG" ]] && BAL_ARGS+=(--config "$CONFIG")
"$PYTHON" "$ROOT/scripts/compute_class_balance.py" "${BAL_ARGS[@]}"

if [[ "$SKIP_TRAIN" != "1" ]]; then
  section "Train CombinedNet"
  TRAIN_ARGS=()
  [[ -n "$CONFIG" ]] && TRAIN_ARGS+=(--config "$CONFIG")
  [[ -n "$EPOCHS" ]] && TRAIN_ARGS+=(--epochs "$EPOCHS")
  [[ "$FRESH_TRAIN" == "1" ]] && TRAIN_ARGS+=(--fresh)
  "$PYTHON" -m src.training.train "${TRAIN_ARGS[@]}"
else
  echo "(Skipping train; SKIP_TRAIN=1)"
fi

BEST_CKPT="$ROOT/artifacts/checkpoints/best.pt"
LATEST_CKPT="$ROOT/artifacts/checkpoints/latest.pt"
if [[ ! -f "$BEST_CKPT" ]] && [[ ! -f "$LATEST_CKPT" ]]; then
  echo "ERROR: No checkpoint found under artifacts/checkpoints/"
  exit 1
fi

if [[ "$SKIP_EVAL" != "1" ]]; then
  section "P6: Evaluate (ACC, F1, AUC)"
  EVAL_ARGS=(--split test)
  [[ -n "$CONFIG" ]] && EVAL_ARGS+=(--config "$CONFIG")
  [[ -f "$BEST_CKPT" ]] && EVAL_ARGS+=(--checkpoint "$BEST_CKPT")
  "$PYTHON" -m src.training.evaluate "${EVAL_ARGS[@]}"
else
  echo "(Skipping eval; SKIP_EVAL=1)"
fi

EXPORT_CKPT="$BEST_CKPT"
[[ -f "$EXPORT_CKPT" ]] || EXPORT_CKPT="$LATEST_CKPT"

if [[ "$SKIP_EXPORT_ONNX" != "1" ]]; then
  section "P7: ONNX export"
  EXPORT_ARGS=()
  [[ -n "$CONFIG" ]] && EXPORT_ARGS+=(--config "$CONFIG")
  EXPORT_ARGS+=(--checkpoint "$EXPORT_CKPT")
  "$PYTHON" "$ROOT/scripts/export_onnx.py" "${EXPORT_ARGS[@]}"
else
  echo "(Skipping ONNX export; SKIP_EXPORT_ONNX=1)"
fi

if [[ "$SKIP_PARITY" != "1" ]]; then
  section "P8: ONNX parity check"
  PARITY_ARGS=()
  [[ -n "$CONFIG" ]] && PARITY_ARGS+=(--config "$CONFIG")
  PARITY_ARGS+=(--checkpoint "$EXPORT_CKPT")
  "$PYTHON" "$ROOT/scripts/parity_check_onnx.py" "${PARITY_ARGS[@]}"
else
  echo "(Skipping parity; SKIP_PARITY=1)"
fi

if [[ "$SKIP_PACKAGE" != "1" ]]; then
  section "Package artifacts"
  BUNDLE="${BUNDLE:-$ROOT/artifacts/pattern_a_bundle.tar.gz}"
  BUNDLE="$BUNDLE" CONFIG="$CONFIG" "$ROOT/scripts/package_artifacts.sh"
fi

if [[ "$PA_ARCHIVE" == "1" ]]; then
  if [[ "$SKIP_PLOTS" != "1" ]]; then
    section "Thesis figures"
    "$PYTHON" "$ARCHIVE_PY" plot --profile "$PROFILE" --root "$ROOT" --run-id "$PA_RUN_ID"
  fi
  section "Finalize run archive"
  "$ROOT/scripts/archive_run.sh" "$PA_RUN_ID"
  section "Thesis snippet"
  "$PYTHON" "$ARCHIVE_PY" snippet --profile "$PROFILE" --root "$ROOT" --run-id "$PA_RUN_ID"
fi

if [[ "$STAGE_ANDROID" != "0" ]]; then
  section "Stage Android assets (P7 → vigidroid/)"
  bash "$REPO_ROOT/Android_Works/stage_early_fusion_dex_manifest.sh"
fi

section "Pattern A pipeline finished"
echo "Manifests:  $MANIFEST_TRAIN"
echo "Best ckpt:  $BEST_CKPT"
echo "Latest:     $LATEST_CKPT"
echo "Metrics:    $ROOT/artifacts/checkpoints/test_results.json"
echo "ONNX bundle: $ROOT/artifacts/export/early_fusion_dex_manifest/"
echo "Dex stats:  $ROOT/artifacts/dex_stats.json"
echo "Failed log: $ROOT/artifacts/failed_apks.log"
echo "Log:        ${PIPELINE_LOG}"
if [[ "$PA_ARCHIVE" == "1" ]]; then
  echo "Run archive: $ROOT/output_archives/$PA_RUN_ID"
  echo "  THESIS_SNIPPET: $ROOT/output_archives/$PA_RUN_ID/THESIS_SNIPPET.md"
fi
if [[ "$STAGE_ANDROID" != "0" ]]; then
  echo "Android assets: $REPO_ROOT/vigidroid/app/src/main/assets/models/early_fusion_dex_manifest/"
fi
echo ""
echo "Done."
