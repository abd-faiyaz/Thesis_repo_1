#!/usr/bin/env bash
# =============================================================================
# run_base_model_1.sh
# End-to-end runner for MSFDroid Base Model 1 (MLP(H)) — Dex header only.
#
# Pipeline (Phases 2 → 6):
#   1. Optional: install Python dependencies
#   2. Optional: verify environment
#   3. Preprocess APKs → dex_header_features.pt
#   4. Train MLP(H) with checkpoint resume
#   5. Evaluate checkpoint (ACC, F1, AUC)
#
# Usage examples:
#   ./run_base_model_1.sh
#   APK_ROOT=/data/apks ./run_base_model_1.sh
#   SKIP_PREPROCESS=1 ./run_base_model_1.sh
#   FRESH_TRAIN=1 EPOCHS=50 ./run_base_model_1.sh
# =============================================================================

# --- Strict bash mode --------------------------------------------------------
# -e  : exit immediately if any command fails
# -u  : treat unset variables as errors
# -o pipefail : fail a pipeline if any stage fails (not just the last)
set -euo pipefail

# --- Resolve project root (folder containing this script) --------------------
# BASH_SOURCE[0] is this script; dirname twice would be wrong — script lives
# in only_base1_model/, so one dirname is enough.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ROOT is the Base Model 1 package root (where src/, config/, artifacts/ live)
ROOT="$SCRIPT_DIR"

# All following commands run from ROOT so relative paths in config work
cd "$ROOT"

# Shared thesis_venv at repo root (see scripts/setup_thesis_venv.sh)
# shellcheck source=/dev/null
source "$ROOT/scripts/activate_thesis_env.sh"

# --- Configurable settings (override via environment variables) --------------

# APK_ROOT: directory tree containing .apk files (benign/ and malware/ subdirs)
# Default matches config/default.yaml → paths.apk_root
APK_ROOT="${APK_ROOT:-$ROOT/data/apks}"

# CONFIG: YAML config file path (optional --config passed to Python modules)
CONFIG="${CONFIG:-$ROOT/config/default.yaml}"

# EPOCHS: training epochs (empty = use value from config/default.yaml)
EPOCHS="${EPOCHS:-}"

# INSTALL_DEPS: if 1, run pip install -r requirements.txt before pipeline
INSTALL_DEPS="${INSTALL_DEPS:-0}"

# VERIFY_SETUP: if 1, run scripts/verify_setup.py before preprocessing
VERIFY_SETUP="${VERIFY_SETUP:-1}"

# SKIP_PREPROCESS: if 1, skip Phase 2 (use existing artifacts/processed/*.pt)
SKIP_PREPROCESS="${SKIP_PREPROCESS:-0}"

# SKIP_TRAIN: if 1, skip Phase 5 training
SKIP_TRAIN="${SKIP_TRAIN:-0}"

# SKIP_EVAL: if 1, skip Phase 6 standalone evaluation
SKIP_EVAL="${SKIP_EVAL:-0}"

# FRESH_TRAIN: if 1, pass --fresh to training (ignore existing checkpoint)
FRESH_TRAIN="${FRESH_TRAIN:-0}"

# PREPROCESS_LIMIT: if set (e.g. 100), only process first N APKs (smoke test)
PREPROCESS_LIMIT="${PREPROCESS_LIMIT:-}"

# --- Python import path ------------------------------------------------------
# Tell Python to import `src.*` from this package root
export PYTHONPATH="${ROOT}${PYTHONPATH:+:$PYTHONPATH}"

# --- Helper: print a visible section banner ----------------------------------
section() {
  echo ""
  echo "============================================================================="
  echo "  $1"
  echo "============================================================================="
}

# --- Step 0: Show configuration ------------------------------------------------
section "Base Model 1 (MLP(H)) — configuration"
echo "ROOT:            $ROOT"
echo "APK_ROOT:        $APK_ROOT"
echo "PYTHON:          $PYTHON"
echo "THESIS_VENV:     ${THESIS_VENV:-<not set>}"
echo "CONFIG:          $CONFIG"
echo "SKIP_PREPROCESS: $SKIP_PREPROCESS"
echo "SKIP_TRAIN:      $SKIP_TRAIN"
echo "SKIP_EVAL:       $SKIP_EVAL"
echo "FRESH_TRAIN:     $FRESH_TRAIN"
echo "INSTALL_DEPS:    $INSTALL_DEPS"
echo "VERIFY_SETUP:    $VERIFY_SETUP"

# --- Step 1 (optional): Install dependencies -----------------------------------
if [[ "$INSTALL_DEPS" == "1" ]]; then
  section "Step 1: Installing dependencies"
  _REQS="$(thesis_all_requirements_path)"
  echo "Using requirements: $_REQS"
  "$PYTHON" -m pip install -r "$_REQS"
else
  echo ""
  echo "(Skipping pip install; set INSTALL_DEPS=1 to install requirements.txt)"
fi

# --- Step 2 (optional): Verify environment -------------------------------------
if [[ "$VERIFY_SETUP" == "1" ]]; then
  section "Step 2: Verifying environment"
  # Checks imports, config load, artifact dirs (Phase 1 smoke test)
  "$PYTHON" "$ROOT/scripts/verify_setup.py"
fi

# --- Step 3: Preprocess APKs (Phase 2) -----------------------------------------
if [[ "$SKIP_PREPROCESS" != "1" ]]; then
  section "Step 3: Preprocessing APKs (Dex header extraction)"

  # Warn if APK folder missing (user may override APK_ROOT on remote machine)
  if [[ ! -d "$APK_ROOT" ]]; then
    echo "WARNING: APK_ROOT does not exist: $APK_ROOT"
    echo "         Create it or set APK_ROOT=/path/to/your/apks before running."
    exit 1
  fi

  # Build optional arguments for preprocess_apks.py
  PREPROCESS_ARGS=(--apk-root "$APK_ROOT")
  if [[ -n "$CONFIG" ]]; then
    PREPROCESS_ARGS+=(--config "$CONFIG")
  fi
  if [[ -n "$PREPROCESS_LIMIT" ]]; then
  # --limit N: process only first N APKs (useful for quick tests)
    PREPROCESS_ARGS+=(--limit "$PREPROCESS_LIMIT")
  fi

  # Unzip APKs in memory, aggregate all classes*.dex headers, save dex_header_features.pt
  "$PYTHON" -m src.preprocessing.preprocess_apks "${PREPROCESS_ARGS[@]}"
else
  echo ""
  echo "(Skipping preprocessing; SKIP_PREPROCESS=1)"
fi

# Path to preprocessed tensor file (must exist before training)
PROCESSED_FILE="$ROOT/artifacts/processed/dex_header_features.pt"
if [[ ! -f "$PROCESSED_FILE" ]]; then
  echo "ERROR: Preprocessed features not found: $PROCESSED_FILE"
  echo "       Run without SKIP_PREPROCESS=1 or copy features to that path."
  exit 1
fi

# --- Step 4: Train MLP(H) (Phases 4–5) -----------------------------------------
if [[ "$SKIP_TRAIN" != "1" ]]; then
  section "Step 4: Training MLP(H)"

  TRAIN_ARGS=()
  if [[ -n "$CONFIG" ]]; then
    TRAIN_ARGS+=(--config "$CONFIG")
  fi
  if [[ -n "$EPOCHS" ]]; then
    # Override training.epochs from YAML
    TRAIN_ARGS+=(--epochs "$EPOCHS")
  fi
  if [[ "$FRESH_TRAIN" == "1" ]]; then
    # Ignore artifacts/checkpoints/latest_checkpoint.pth
    TRAIN_ARGS+=(--fresh)
  fi

  # SGD + BCELoss; tqdm progress; saves checkpoint each epoch; prints ACC/F1/AUC
  "$PYTHON" -m src.training.train "${TRAIN_ARGS[@]}"
else
  echo ""
  echo "(Skipping training; SKIP_TRAIN=1)"
fi

# Checkpoint path used by training and evaluation
CHECKPOINT="$ROOT/artifacts/checkpoints/latest_checkpoint.pth"
if [[ ! -f "$CHECKPOINT" ]]; then
  echo "ERROR: Training checkpoint not found: $CHECKPOINT"
  echo "       Train first or place a valid latest_checkpoint.pth there."
  exit 1
fi

# --- Step 5: Standalone evaluation (Phase 6) -----------------------------------
if [[ "$SKIP_EVAL" != "1" ]]; then
  section "Step 5: Evaluation (ACC, F1, AUC)"

  EVAL_ARGS=(--split val)
  if [[ -n "$CONFIG" ]]; then
    EVAL_ARGS+=(--config "$CONFIG")
  fi
  EVAL_ARGS+=(--checkpoint "$CHECKPOINT")

  # Load checkpoint and report sklearn metrics on validation split
  "$PYTHON" -m src.training.evaluate "${EVAL_ARGS[@]}"
else
  echo ""
  echo "(Skipping evaluation; SKIP_EVAL=1)"
fi

# --- Done ---------------------------------------------------------------------
section "Base Model 1 pipeline finished"
echo "Processed features: $PROCESSED_FILE"
echo "Checkpoint:         $CHECKPOINT"
echo "Failed APK log:     $ROOT/artifacts/failed_apks.log (if any failures)"
echo ""
echo "Done."
