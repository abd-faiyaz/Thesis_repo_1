#!/usr/bin/env bash
# =============================================================================
# run_pattern_b.sh — End-to-end Pattern B (Dual-Branch Merge) pipeline
#
# Phases 2 → 6:
#   1. Optional: install dependencies
#   2. Optional: verify environment
#   3. Preprocess APKs → per-APK shards + manifests
#   4. Compute class balance → artifacts/class_balance.json
#   5. Train DualBranchNet (resume-safe)
#   6. Evaluate best/latest checkpoint (ACC, F1, AUC)
#   7. Optional: package portable artifacts tarball
#
# Usage:
#   ./run_pattern_b.sh
#   APK_ROOT=/data/apks ./run_pattern_b.sh
#   SKIP_PREPROCESS=1 ./run_pattern_b.sh
#   PREPROCESS_LIMIT=200 EPOCHS=2 ./run_pattern_b.sh   # smoke test
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$SCRIPT_DIR"
cd "$ROOT"

# shellcheck source=/dev/null
source "$ROOT/scripts/activate_thesis_env.sh"

APK_ROOT="${APK_ROOT:-$ROOT/data/apks}"
CONFIG="${CONFIG:-$ROOT/config/default.yaml}"
EPOCHS="${EPOCHS:-}"
INSTALL_DEPS="${INSTALL_DEPS:-0}"
VERIFY_SETUP="${VERIFY_SETUP:-1}"
SKIP_PREPROCESS="${SKIP_PREPROCESS:-0}"
SKIP_TRAIN="${SKIP_TRAIN:-0}"
SKIP_EVAL="${SKIP_EVAL:-0}"
SKIP_PACKAGE="${SKIP_PACKAGE:-0}"
FRESH_TRAIN="${FRESH_TRAIN:-0}"
PREPROCESS_LIMIT="${PREPROCESS_LIMIT:-}"
EXTRACT_LIMIT="${EXTRACT_LIMIT:-}"
PIPELINE_LOG="${PIPELINE_LOG:-$ROOT/artifacts/pipeline.log}"

export PYTHONPATH="${ROOT}${PYTHONPATH:+:$PYTHONPATH}"

section() {
  echo ""
  echo "============================================================================="
  echo "  $1"
  echo "============================================================================="
}

mkdir -p "$(dirname "$PIPELINE_LOG")"
exec > >(tee -a "$PIPELINE_LOG") 2>&1

section "Pattern B (Dual-Branch Merge) — configuration"
echo "ROOT:             $ROOT"
echo "APK_ROOT:         $APK_ROOT"
echo "PYTHON:           $PYTHON"
echo "THESIS_VENV:      ${THESIS_VENV:-<not set>}"
echo "CONFIG:           $CONFIG"
echo "SKIP_PREPROCESS:  $SKIP_PREPROCESS"
echo "SKIP_TRAIN:       $SKIP_TRAIN"
echo "SKIP_EVAL:        $SKIP_EVAL"
echo "SKIP_PACKAGE:     $SKIP_PACKAGE"
echo "FRESH_TRAIN:      $FRESH_TRAIN"
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
  section "Preprocess (scan → lexicon → norm → shards)"
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

  EXTRACT_ARGS=(--split both)
  [[ -n "$CONFIG" ]] && EXTRACT_ARGS+=(--config "$CONFIG")
  [[ -n "$EXTRACT_LIMIT" ]] && EXTRACT_ARGS+=(--limit "$EXTRACT_LIMIT")
  "$PYTHON" -m src.preprocessing.extract_to_cache "${EXTRACT_ARGS[@]}"
else
  echo "(Skipping preprocess; SKIP_PREPROCESS=1)"
fi

MANIFEST_TRAIN="$ROOT/artifacts/processed/manifest_train.json"
MANIFEST_VAL="$ROOT/artifacts/processed/manifest_val.json"
if [[ ! -f "$MANIFEST_TRAIN" ]] || [[ ! -f "$MANIFEST_VAL" ]]; then
  echo "ERROR: Missing manifests. Run preprocessing first."
  exit 1
fi

section "Class balance (train split)"
BAL_ARGS=()
[[ -n "$CONFIG" ]] && BAL_ARGS+=(--config "$CONFIG")
"$PYTHON" "$ROOT/scripts/compute_class_balance.py" "${BAL_ARGS[@]}"

if [[ "$SKIP_TRAIN" != "1" ]]; then
  section "Train DualBranchNet"
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
  section "Evaluate (ACC, F1, AUC)"
  EVAL_ARGS=(--split val)
  [[ -n "$CONFIG" ]] && EVAL_ARGS+=(--config "$CONFIG")
  [[ -f "$BEST_CKPT" ]] && EVAL_ARGS+=(--checkpoint "$BEST_CKPT")
  "$PYTHON" -m src.training.evaluate "${EVAL_ARGS[@]}"
else
  echo "(Skipping eval; SKIP_EVAL=1)"
fi

if [[ "$SKIP_PACKAGE" != "1" ]]; then
  section "Package artifacts"
  BUNDLE="${BUNDLE:-$ROOT/artifacts/pattern_b_bundle.tar.gz}"
  BUNDLE="$BUNDLE" CONFIG="$CONFIG" "$ROOT/scripts/package_artifacts.sh"
fi

section "Pattern B pipeline finished"
echo "Manifests:  $MANIFEST_TRAIN"
echo "Best ckpt:  $BEST_CKPT"
echo "Latest:     $LATEST_CKPT"
echo "Failed log: $ROOT/artifacts/failed_apks.log"
echo "Log:        $PIPELINE_LOG"
echo ""
echo "Done."
