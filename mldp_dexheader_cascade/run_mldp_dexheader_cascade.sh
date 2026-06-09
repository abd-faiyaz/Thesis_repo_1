#!/usr/bin/env bash
# =============================================================================
# run_mldp_dexheader_cascade.sh
# End-to-end runner for MLDP permissions + Dex header cascade hybrid.
#
# Pipeline:
#   P0  verify_setup (+ optional pip install)
#   P1  index_dataset (APK corpus + temporal splits)
#   P2  preprocess APKs → MLDP freeze S, dex min/max, feature shards
#   P3  verify_dataloader
#   P4  verify_model (architecture smoke)
#   P5  train Mode A fused MLP + Mode B Stage-1 (+ ablations / paper baselines)
#   P6  evaluate on 2022+2023 test split (Mode A + Mode B cascade)
#   P7  export ONNX bundle (Mode A + Mode B Stage-1; Stage-2 = deployed mlp_header copy)
#   P8  PyTorch vs ONNX parity on parity_samples
#   +   figures, archive finalize, THESIS_SNIPPET (default on; MDH_ARCHIVE=0 or SKIP_ARCHIVE=1 to disable)
#   +   Android asset staging by default (set STAGE_ANDROID=0 to skip)
#
# Usage:
#   ./run_mldp_dexheader_cascade.sh
#   APK_ROOT=/mnt/Files/thesis_full_dataset ./run_mldp_dexheader_cascade.sh
#   SKIP_PREPROCESS=1 ./run_mldp_dexheader_cascade.sh
#   QUICK=1 PREPROCESS_LIMIT=200 ./run_mldp_dexheader_cascade.sh
#   MDH_ARCHIVE=1 MDH_RUN_ID=run_20260607_mdh ./run_mldp_dexheader_cascade.sh
#
# Fresh full run (wipe artifacts first):
#   rm -rf artifacts/processed artifacts/checkpoints artifacts/metrics
#   rm -rf artifacts/export artifacts/manifests artifacts/splits
#   rm -f artifacts/failed_apks.log artifacts/failed_index.log
#   FRESH_TRAIN=1 MDH_ARCHIVE=1 ./run_mldp_dexheader_cascade.sh
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$SCRIPT_DIR"
REPO_ROOT="$(cd "$ROOT/.." && pwd)"
cd "$ROOT"

# shellcheck source=/dev/null
source "$ROOT/scripts/activate_thesis_env.sh"
PYTHON="${PYTHON:-python3}"
export PYTHON
export PYTHONPATH="${ROOT}${PYTHONPATH:+:$PYTHONPATH}"

ARCHIVE_PY="$REPO_ROOT/scripts/thesis_run_archive.py"
PROFILE="mldp_dexheader_cascade"

APK_ROOT="${APK_ROOT:-}"
CONFIG="${CONFIG:-$ROOT/config/default.yaml}"
INSTALL_DEPS="${INSTALL_DEPS:-0}"
VERIFY_SETUP="${VERIFY_SETUP:-1}"
SKIP_INDEX="${SKIP_INDEX:-0}"
SKIP_PREPROCESS="${SKIP_PREPROCESS:-0}"
SKIP_VERIFY_DATALOADER="${SKIP_VERIFY_DATALOADER:-0}"
SKIP_VERIFY_MODEL="${SKIP_VERIFY_MODEL:-0}"
SKIP_TRAIN="${SKIP_TRAIN:-0}"
SKIP_EVAL="${SKIP_EVAL:-0}"
SKIP_EXPORT_ONNX="${SKIP_EXPORT_ONNX:-0}"
SKIP_PARITY="${SKIP_PARITY:-0}"
SKIP_ARCHIVE="${SKIP_ARCHIVE:-0}"
SKIP_PLOTS="${SKIP_PLOTS:-0}"
STAGE_ANDROID="${STAGE_ANDROID:-1}"
FRESH_TRAIN="${FRESH_TRAIN:-0}"
PREPROCESS_LIMIT="${PREPROCESS_LIMIT:-}"
EPOCHS="${EPOCHS:-}"
QUICK="${QUICK:-0}"

MDH_ARCHIVE="${MDH_ARCHIVE:-1}"
MDH_RUN_ID="${MDH_RUN_ID:-}"
if [[ "$SKIP_ARCHIVE" == "1" ]]; then
  MDH_ARCHIVE=0
fi

MODE_A_CKPT="$ROOT/artifacts/checkpoints/mode_a_best.pt"
STAGE1_CKPT="$ROOT/artifacts/checkpoints/stage1_best.pt"
EXPORT_DIR="$ROOT/artifacts/export/mldp_dexheader_cascade"
METRICS_DIR="$ROOT/artifacts/metrics"

if [[ "$MDH_ARCHIVE" == "1" ]]; then
  if [[ -z "$MDH_RUN_ID" ]]; then
    MDH_RUN_ID="run_$(date +%Y%m%d_%H%M%S)_mdh"
  fi
  export MDH_RUN_ID
  ARCHIVE_DIR="$ROOT/output_archives/$MDH_RUN_ID"
  "$PYTHON" "$ARCHIVE_PY" bootstrap \
    --profile "$PROFILE" \
    --root "$ROOT" \
    --run-id "$MDH_RUN_ID" \
    --config "$CONFIG" \
    ${APK_ROOT:+--apk-root "$APK_ROOT"}
  exec > >(tee -a "$ARCHIVE_DIR/logs/pipeline_full.log") 2>&1
fi

section() {
  echo ""
  echo "============================================================================="
  echo "  $1"
  echo "============================================================================="
}

_cfg_yaml() {
  "$PYTHON" -c "import yaml; c=yaml.safe_load(open('$CONFIG')); print($1)" 2>/dev/null || echo "?"
}

section "MLDP + Dex header cascade — configuration"
echo "ROOT:                $ROOT"
echo "APK_ROOT:            ${APK_ROOT:-$(_cfg_yaml "c.get('paths',{}).get('apk_root','<from config>')")}"
echo "PYTHON:              $PYTHON"
echo "THESIS_VENV:         ${THESIS_VENV:-<not set>}"
echo "CONFIG:              $CONFIG"
echo "SKIP_INDEX:          $SKIP_INDEX"
echo "SKIP_PREPROCESS:     $SKIP_PREPROCESS"
echo "SKIP_VERIFY_DL:      $SKIP_VERIFY_DATALOADER"
echo "SKIP_VERIFY_MODEL:   $SKIP_VERIFY_MODEL"
echo "SKIP_TRAIN:          $SKIP_TRAIN"
echo "SKIP_EVAL:           $SKIP_EVAL"
echo "SKIP_EXPORT:         $SKIP_EXPORT_ONNX"
echo "SKIP_PARITY:         $SKIP_PARITY"
echo "SKIP_ARCHIVE:        $SKIP_ARCHIVE"
echo "SKIP_PLOTS:          $SKIP_PLOTS"
echo "STAGE_ANDROID:       $STAGE_ANDROID"
echo "FRESH_TRAIN:         $FRESH_TRAIN"
echo "QUICK:               $QUICK"
echo "PREPROCESS_LIMIT:    ${PREPROCESS_LIMIT:-<none>}"
echo "MDH_ARCHIVE:         $MDH_ARCHIVE"
echo "MDH_RUN_ID:          ${MDH_RUN_ID:-<auto when MDH_ARCHIVE=1>}"
echo "Train years:         $(_cfg_yaml "c.get('splits',{}).get('train_years','?')")"
echo "Test years:          $(_cfg_yaml "c.get('splits',{}).get('test_years','?')")"

if [[ "$INSTALL_DEPS" == "1" ]]; then
  section "Installing dependencies"
  REQ="$REPO_ROOT/requirements-thesis-all.txt"
  if [[ ! -f "$REQ" ]]; then
    REQ="$ROOT/requirements.txt"
  fi
  echo "Using requirements: $REQ"
  "$PYTHON" -m pip install -r "$REQ"
else
  echo ""
  echo "(Skipping pip install; set INSTALL_DEPS=1 to install requirements)"
fi

if [[ "$VERIFY_SETUP" == "1" ]]; then
  section "P0: Verifying environment"
  "$PYTHON" "$ROOT/scripts/verify_setup.py"
fi

if [[ "$SKIP_INDEX" != "1" ]]; then
  section "P1: Indexing APK corpus"
  INDEX_ARGS=(--config "$CONFIG")
  [[ -n "$APK_ROOT" ]] && INDEX_ARGS+=(--apk-root "$APK_ROOT")
  [[ -n "$PREPROCESS_LIMIT" ]] && INDEX_ARGS+=(--limit "$PREPROCESS_LIMIT")
  "$PYTHON" "$ROOT/scripts/index_dataset.py" "${INDEX_ARGS[@]}"
else
  echo "(Skipping index; SKIP_INDEX=1)"
fi

INDEX_CSV="$ROOT/artifacts/manifests/apk_index.csv"
if [[ ! -f "$INDEX_CSV" ]]; then
  echo "ERROR: APK index not found: $INDEX_CSV"
  echo "       Run without SKIP_INDEX=1 or ensure Shared_pipeline_Files manifest exists."
  exit 1
fi

if [[ "$SKIP_PREPROCESS" != "1" ]]; then
  section "P2: Preprocessing (MLDP S + dex headers + vectorize)"
  PRE_ARGS=(--config "$CONFIG")
  [[ -n "$PREPROCESS_LIMIT" ]] && PRE_ARGS+=(--limit "$PREPROCESS_LIMIT")
  "$PYTHON" -m src.preprocessing.preprocess_apks "${PRE_ARGS[@]}"
else
  echo "(Skipping preprocessing; SKIP_PREPROCESS=1)"
fi

PROCESSED_DIR="$ROOT/artifacts/processed"
if [[ ! -f "$PROCESSED_DIR/features_train.pt" ]] || [[ ! -f "$PROCESSED_DIR/features_test.pt" ]]; then
  echo "ERROR: Missing feature shards under $PROCESSED_DIR"
  echo "       Expected features_train.pt and features_test.pt"
  exit 1
fi

"$PYTHON" "$ARCHIVE_PY" export-corpus-stats \
  --profile "$PROFILE" \
  --root "$ROOT" \
  ${MDH_RUN_ID:+--run-id "$MDH_RUN_ID"}

if [[ "$SKIP_VERIFY_DATALOADER" != "1" ]]; then
  section "P3: Verifying DataLoaders and split"
  "$PYTHON" "$ROOT/scripts/verify_dataloader.py" --config "$CONFIG"
else
  echo "(Skipping DataLoader verify; SKIP_VERIFY_DATALOADER=1)"
fi

if [[ "$SKIP_VERIFY_MODEL" != "1" ]]; then
  section "P4: Verifying model architecture"
  "$PYTHON" "$ROOT/scripts/verify_model.py"
else
  echo "(Skipping model verify; SKIP_VERIFY_MODEL=1)"
fi

if [[ "$SKIP_TRAIN" != "1" ]]; then
  section "P5: Training Mode A + Mode B Stage-1"
  if [[ "$FRESH_TRAIN" == "1" ]]; then
    rm -f "$MODE_A_CKPT" "$STAGE1_CKPT"
    echo "FRESH_TRAIN=1 → removed checkpoints under artifacts/checkpoints/"
  fi
  TRAIN_ARGS=(--config "$CONFIG")
  [[ -n "$EPOCHS" ]] && TRAIN_ARGS+=(--epochs "$EPOCHS")
  if [[ "$QUICK" == "1" ]]; then
    TRAIN_ARGS+=(--epochs 8 --skip-baselines --skip-ablations)
    echo "QUICK=1 → 8 epochs, skipping sklearn baselines and ablations"
  fi
  "$PYTHON" -m src.training.train "${TRAIN_ARGS[@]}"
else
  echo "(Skipping training; SKIP_TRAIN=1)"
fi

if [[ ! -f "$MODE_A_CKPT" ]]; then
  echo "ERROR: Mode A checkpoint not found: $MODE_A_CKPT"
  exit 1
fi
if [[ ! -f "$STAGE1_CKPT" ]]; then
  echo "ERROR: Stage-1 checkpoint not found: $STAGE1_CKPT"
  exit 1
fi

if [[ "$SKIP_EVAL" != "1" ]]; then
  section "P6: Evaluation (temporal test split 2022+2023)"
  "$PYTHON" -m src.training.evaluate \
    --config "$CONFIG" \
    --metrics-out "$METRICS_DIR/test_results.json"
else
  echo "(Skipping evaluation; SKIP_EVAL=1)"
fi

if [[ "$SKIP_EXPORT_ONNX" != "1" ]]; then
  section "P7: ONNX export (Mode A + Mode B bundle)"
  EXPORT_ARGS=(
    --config "$CONFIG"
    --mode-a-checkpoint "$MODE_A_CKPT"
    --stage1-checkpoint "$STAGE1_CKPT"
  )
  if [[ "$STAGE_ANDROID" == "1" ]]; then
    EXPORT_ARGS+=(--deploy-vigidroid)
  fi
  "$PYTHON" "$ROOT/scripts/export_onnx.py" "${EXPORT_ARGS[@]}"
else
  echo "(Skipping ONNX export; SKIP_EXPORT_ONNX=1)"
fi

if [[ ! -f "$EXPORT_DIR/mode_a/model.onnx" ]]; then
  echo "ERROR: Mode A ONNX not found: $EXPORT_DIR/mode_a/model.onnx"
  exit 1
fi

if [[ "$SKIP_PARITY" != "1" ]]; then
  section "P8: ONNX parity check"
  "$PYTHON" -m src.training.parity_onnx \
    --config "$CONFIG" \
    --mode-a-checkpoint "$MODE_A_CKPT" \
    --stage1-checkpoint "$STAGE1_CKPT"
else
  echo "(Skipping parity; SKIP_PARITY=1)"
fi

if [[ "$MDH_ARCHIVE" == "1" ]]; then
  if [[ "$SKIP_PLOTS" != "1" ]]; then
    section "Thesis figures"
    "$PYTHON" "$ARCHIVE_PY" plot \
      --profile "$PROFILE" \
      --root "$ROOT" \
      --run-id "$MDH_RUN_ID"
  else
    echo "(Skipping figures; SKIP_PLOTS=1)"
  fi
  section "Finalize run archive"
  "$ROOT/scripts/archive_run.sh" "$MDH_RUN_ID"
  section "Thesis snippet"
  "$PYTHON" "$ARCHIVE_PY" snippet \
    --profile "$PROFILE" \
    --root "$ROOT" \
    --run-id "$MDH_RUN_ID"
fi

if [[ "$STAGE_ANDROID" != "0" ]]; then
  section "Stage Android assets (P7 → vigidroid/)"
  bash "$REPO_ROOT/Android_Works/stage_mldp_dexheader_cascade.sh"
fi

section "MLDP + Dex header cascade pipeline finished"
echo "Index:           $INDEX_CSV"
echo "Processed:       $PROCESSED_DIR"
echo "Mode A ckpt:     $MODE_A_CKPT"
echo "Stage-1 ckpt:    $STAGE1_CKPT"
echo "Test metrics:    $METRICS_DIR/test_results.json"
echo "Parity report:   $METRICS_DIR/parity_report.json"
echo "ONNX bundle:     $EXPORT_DIR/"
echo "Failed APK log:  $ROOT/artifacts/failed_apks.log (if any)"
if [[ "$MDH_ARCHIVE" == "1" ]]; then
  echo "Run archive:     $ROOT/output_archives/$MDH_RUN_ID"
  echo "  THESIS_SNIPPET: $ROOT/output_archives/$MDH_RUN_ID/THESIS_SNIPPET.md"
fi
if [[ "$STAGE_ANDROID" != "0" ]]; then
  echo "Android assets:  $REPO_ROOT/vigidroid/app/src/main/assets/models/mldp_dexheader_cascade/"
  echo "Next on phone:   bash Android_Works/run_mldp_dexheader_a4.sh"
fi
echo ""
echo "Done."
