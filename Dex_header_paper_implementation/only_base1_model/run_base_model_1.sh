#!/usr/bin/env bash
# =============================================================================
# run_base_model_1.sh
# End-to-end runner for MSFDroid Base Model 1 (MLP(H)) — Dex header only.
#
# Pipeline (aligns with Pipeline_full_concept.html P0–P8 for Python / D3):
#   P0  verify_setup (+ optional pip install)
#   P2  preprocess APKs → dex_header_features.pt + corpus stats
#   P3  verify_dataloader (temporal or random split from config)
#   P5  train MLP(H) with checkpoint resume
#   P6  evaluate on val/test split (ACC, F1, AUC)
#   P7  export ONNX bundle (artifacts/export/mlp_header/)
#   P8  PyTorch vs ONNX parity on parity_samples
#   +   figures, archive finalize, THESIS_SNIPPET when BM1_ARCHIVE=1
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

# SKIP_VERIFY_DATALOADER: if 1, skip P3 DataLoader / split check
SKIP_VERIFY_DATALOADER="${SKIP_VERIFY_DATALOADER:-0}"

# SKIP_EXPORT_ONNX: if 1, skip P7 ONNX export bundle
SKIP_EXPORT_ONNX="${SKIP_EXPORT_ONNX:-0}"

# SKIP_PARITY: if 1, skip P8 PyTorch vs ONNX parity
SKIP_PARITY="${SKIP_PARITY:-0}"

# SKIP_PLOTS: if 1, skip thesis figures (requires BM1_ARCHIVE=1)
SKIP_PLOTS="${SKIP_PLOTS:-0}"

# FRESH_TRAIN: if 1, pass --fresh to training (ignore existing checkpoint)
FRESH_TRAIN="${FRESH_TRAIN:-0}"

# PREPROCESS_LIMIT: if set (e.g. 100), only process first N APKs (smoke test)
PREPROCESS_LIMIT="${PREPROCESS_LIMIT:-}"

# BM1_ARCHIVE: if 1, tee logs + mirror JSON metrics to output_archives/${BM1_RUN_ID}/
BM1_ARCHIVE="${BM1_ARCHIVE:-0}"
BM1_RUN_ID="${BM1_RUN_ID:-}"

# --- Python import path ------------------------------------------------------
# Tell Python to import `src.*` from this package root
export PYTHONPATH="${ROOT}${PYTHONPATH:+:$PYTHONPATH}"

# --- Optional run archive (output_archives/) ---------------------------------
if [[ "$BM1_ARCHIVE" == "1" ]]; then
  if [[ -z "$BM1_RUN_ID" ]]; then
    BM1_RUN_ID="run_$(date +%Y%m%d_%H%M%S)_bm1"
  fi
  export BM1_RUN_ID
  ARCHIVE_DIR="$ROOT/output_archives/$BM1_RUN_ID"
  mkdir -p "$ARCHIVE_DIR"/{logs,metrics,corpus_stats,figures,config,export,parity}
  echo "$BM1_RUN_ID" > "$ROOT/output_archives/LATEST_RUN.txt"
  cp "$CONFIG" "$ARCHIVE_DIR/config/default.yaml.snapshot"
  exec > >(tee -a "$ARCHIVE_DIR/logs/pipeline_full.log") 2>&1
fi

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
echo "SKIP_VERIFY_DL:  $SKIP_VERIFY_DATALOADER"
echo "SKIP_EXPORT:     $SKIP_EXPORT_ONNX"
echo "SKIP_PARITY:     $SKIP_PARITY"
echo "SKIP_PLOTS:      $SKIP_PLOTS"
echo "FRESH_TRAIN:     $FRESH_TRAIN"
echo "INSTALL_DEPS:    $INSTALL_DEPS"
echo "VERIFY_SETUP:    $VERIFY_SETUP"
echo "BM1_ARCHIVE:     $BM1_ARCHIVE"
echo "BM1_RUN_ID:      ${BM1_RUN_ID:-<auto when BM1_ARCHIVE=1>}"
SPLIT_MODE="$("$PYTHON" -c "import yaml; c=yaml.safe_load(open('$CONFIG')); print(c.get('preprocessing',{}).get('split_mode','?'))")"
TRAIN_YEARS="$("$PYTHON" -c "import yaml; c=yaml.safe_load(open('$CONFIG')); print(c.get('preprocessing',{}).get('train_years','?'))")"
TEST_YEARS="$("$PYTHON" -c "import yaml; c=yaml.safe_load(open('$CONFIG')); p=c.get('preprocessing',{}); print(p.get('test_years', p.get('val_years','?')))")"
VAL_FRAC="$("$PYTHON" -c "import yaml; c=yaml.safe_load(open('$CONFIG')); p=c.get('preprocessing',{}); print(p.get('val_fraction', c.get('data',{}).get('val_fraction','?')))")"
echo "SPLIT_MODE:      $SPLIT_MODE (train years $TRAIN_YEARS, test years $TEST_YEARS, val_fraction $VAL_FRAC)"

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

# Corpus stats JSON (labels, dex counts, year folders)
"$PYTHON" "$ROOT/scripts/export_corpus_stats.py" ${CONFIG:+--config "$CONFIG"}

# --- Step 3b: Verify DataLoaders / split (P3) ----------------------------------
if [[ "$SKIP_VERIFY_DATALOADER" != "1" ]]; then
  section "Step 3b: Verifying DataLoaders and split"
  "$PYTHON" "$ROOT/scripts/verify_dataloader.py" ${CONFIG:+--config "$CONFIG"}
else
  echo ""
  echo "(Skipping DataLoader verify; SKIP_VERIFY_DATALOADER=1)"
fi

# --- Step 4: Train MLP(H) (P5) -------------------------------------------------
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

# --- Step 5: Standalone evaluation (P6) ----------------------------------------
if [[ "$SKIP_EVAL" != "1" ]]; then
  section "Step 5: Evaluation (ACC, F1, AUC)"

  EVAL_ARGS=(--split test --metrics-out "$ROOT/artifacts/metrics/test_results.json")
  if [[ -n "$CONFIG" ]]; then
    EVAL_ARGS+=(--config "$CONFIG")
  fi
  EVAL_ARGS+=(--checkpoint "$CHECKPOINT")

  # Final metrics on temporal holdout (2022+2023); val split is only for training monitoring
  "$PYTHON" -m src.training.evaluate "${EVAL_ARGS[@]}"
else
  echo ""
  echo "(Skipping evaluation; SKIP_EVAL=1)"
fi

# --- Step 6: ONNX export bundle (P7) -------------------------------------------
if [[ "$SKIP_EXPORT_ONNX" != "1" ]]; then
  section "Step 6: ONNX export (P7)"
  EXPORT_ARGS=()
  [[ -n "$CONFIG" ]] && EXPORT_ARGS+=(--config "$CONFIG")
  EXPORT_ARGS+=(--checkpoint "$CHECKPOINT")
  "$PYTHON" "$ROOT/scripts/export_onnx.py" "${EXPORT_ARGS[@]}"
else
  echo ""
  echo "(Skipping ONNX export; SKIP_EXPORT_ONNX=1)"
fi

# --- Step 7: PyTorch vs ONNX parity (P8) ---------------------------------------
if [[ "$SKIP_PARITY" != "1" ]]; then
  section "Step 7: ONNX parity check (P8)"
  PARITY_ARGS=()
  [[ -n "$CONFIG" ]] && PARITY_ARGS+=(--config "$CONFIG")
  PARITY_ARGS+=(--checkpoint "$CHECKPOINT")
  "$PYTHON" "$ROOT/scripts/parity_check_onnx.py" "${PARITY_ARGS[@]}"
else
  echo ""
  echo "(Skipping parity check; SKIP_PARITY=1)"
fi

# --- Step 8: Figures + archive finalize (when BM1_ARCHIVE=1) -------------------
if [[ "$BM1_ARCHIVE" == "1" ]]; then
  if [[ "$SKIP_PLOTS" != "1" ]]; then
    section "Step 8a: Thesis figures"
    PLOT_ARGS=(--checkpoint "$CHECKPOINT")
    [[ -n "$CONFIG" ]] && PLOT_ARGS+=(--config "$CONFIG")
    "$PYTHON" "$ROOT/scripts/plot_bm1_results.py" "${PLOT_ARGS[@]}"
  else
    echo ""
    echo "(Skipping figures; SKIP_PLOTS=1)"
  fi

  section "Step 8b: Finalize run archive"
  "$ROOT/scripts/archive_run.sh" "$BM1_RUN_ID"

  section "Step 8c: Thesis snippet"
  "$PYTHON" "$ROOT/scripts/generate_thesis_snippet.py"
else
  echo ""
  echo "(Skipping figures/archive finalize/thesis snippet; set BM1_ARCHIVE=1 to enable)"
fi

# --- Done ---------------------------------------------------------------------
section "Base Model 1 pipeline finished"
echo "Processed features: $PROCESSED_FILE"
echo "Checkpoint:         $CHECKPOINT"
echo "Metrics dir:        $ROOT/artifacts/metrics"
echo "ONNX bundle:        $ROOT/artifacts/export/mlp_header/"
echo "Splits:             $ROOT/artifacts/splits/ (train.txt, val.txt, test.txt)"
echo "Failed APK log:     $ROOT/artifacts/failed_apks.log (if any failures)"
if [[ "$BM1_ARCHIVE" == "1" ]]; then
  echo "Run archive:        $ROOT/output_archives/$BM1_RUN_ID"
  echo "  RUN_MANIFEST:     $ROOT/output_archives/$BM1_RUN_ID/RUN_MANIFEST.json"
  echo "  THESIS_SNIPPET:   $ROOT/output_archives/$BM1_RUN_ID/THESIS_SNIPPET.md"
fi
echo ""
echo "Done."
