#!/usr/bin/env bash
# =============================================================================
# run_broadcast_mldp_hybrid.sh
# End-to-end runner for Broadcast + MLDP Hybrid (manifest permissions + receivers).
#
# Pipeline:
#   P0  build_system_actions + verify_setup
#   P1  index_dataset (APK manifest + splits)
#   P2  preprocess APKs → feature shards + frozen vocabs
#   P3  verify_dataloader
#   P4  verify_model (architecture + sklearn baselines smoke)
#   P5  train tiny MLP (+ ablations / paper baselines)
#   P6  evaluate on test split (ACC, F1, AUC)
#   P7  export ONNX bundle (artifacts/export/broadcast_mldp_hybrid/)
#   P8  PyTorch vs ONNX parity
#   +   figures, archive finalize, THESIS_SNIPPET when BMH_ARCHIVE=1
#
# Usage:
#   ./run_broadcast_mldp_hybrid.sh
#   APK_ROOT=/data/apks ./run_broadcast_mldp_hybrid.sh
#   SKIP_PREPROCESS=1 ./run_broadcast_mldp_hybrid.sh
#   QUICK=1 BMH_ARCHIVE=1 ./run_broadcast_mldp_hybrid.sh
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
PROFILE="broadcast_mldp_hybrid"

APK_ROOT="${APK_ROOT:-}"
CONFIG="${CONFIG:-$ROOT/config/default.yaml}"
INSTALL_DEPS="${INSTALL_DEPS:-0}"
VERIFY_SETUP="${VERIFY_SETUP:-1}"
SKIP_BUILD_SYSTEM_ACTIONS="${SKIP_BUILD_SYSTEM_ACTIONS:-0}"
SKIP_INDEX="${SKIP_INDEX:-0}"
SKIP_PREPROCESS="${SKIP_PREPROCESS:-0}"
SKIP_VERIFY_DATALOADER="${SKIP_VERIFY_DATALOADER:-0}"
SKIP_VERIFY_MODEL="${SKIP_VERIFY_MODEL:-0}"
SKIP_TRAIN="${SKIP_TRAIN:-0}"
SKIP_EVAL="${SKIP_EVAL:-0}"
SKIP_EXPORT_ONNX="${SKIP_EXPORT_ONNX:-0}"
SKIP_PARITY="${SKIP_PARITY:-0}"
SKIP_PLOTS="${SKIP_PLOTS:-0}"
FRESH_TRAIN="${FRESH_TRAIN:-0}"
PREPROCESS_LIMIT="${PREPROCESS_LIMIT:-}"
EPOCHS="${EPOCHS:-}"
QUICK="${QUICK:-0}"

BMH_ARCHIVE="${BMH_ARCHIVE:-0}"
BMH_RUN_ID="${BMH_RUN_ID:-}"

if [[ "$BMH_ARCHIVE" == "1" ]]; then
  if [[ -z "$BMH_RUN_ID" ]]; then
    BMH_RUN_ID="run_$(date +%Y%m%d_%H%M%S)_bmh"
  fi
  export BMH_RUN_ID
  ARCHIVE_DIR="$ROOT/output_archives/$BMH_RUN_ID"
  "$PYTHON" "$ARCHIVE_PY" bootstrap \
    --profile "$PROFILE" \
    --root "$ROOT" \
    --run-id "$BMH_RUN_ID" \
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

section "Broadcast + MLDP Hybrid — configuration"
echo "ROOT:                $ROOT"
echo "APK_ROOT:            ${APK_ROOT:-<from config>}"
echo "CONFIG:              $CONFIG"
echo "SKIP_INDEX:          $SKIP_INDEX"
echo "SKIP_PREPROCESS:     $SKIP_PREPROCESS"
echo "SKIP_VERIFY_DL:      $SKIP_VERIFY_DATALOADER"
echo "SKIP_VERIFY_MODEL:   $SKIP_VERIFY_MODEL"
echo "SKIP_TRAIN:          $SKIP_TRAIN"
echo "SKIP_EVAL:           $SKIP_EVAL"
echo "SKIP_EXPORT:         $SKIP_EXPORT_ONNX"
echo "SKIP_PARITY:         $SKIP_PARITY"
echo "SKIP_PLOTS:          $SKIP_PLOTS"
echo "FRESH_TRAIN:         $FRESH_TRAIN"
echo "QUICK:               $QUICK"
echo "BMH_ARCHIVE:         $BMH_ARCHIVE"
echo "BMH_RUN_ID:          ${BMH_RUN_ID:-<auto when BMH_ARCHIVE=1>}"

if [[ "$INSTALL_DEPS" == "1" ]]; then
  section "Installing dependencies"
  REQ="$REPO_ROOT/requirements-thesis-all.txt"
  if [[ ! -f "$REQ" ]]; then
    REQ="$ROOT/requirements.txt"
  fi
  "$PYTHON" -m pip install -r "$REQ"
fi

if [[ "$SKIP_BUILD_SYSTEM_ACTIONS" != "1" ]]; then
  section "P0a: Build system_actions.json"
  "$PYTHON" "$ROOT/scripts/build_system_actions.py"
else
  echo "(Skipping build_system_actions; SKIP_BUILD_SYSTEM_ACTIONS=1)"
fi

if [[ "$VERIFY_SETUP" == "1" ]]; then
  section "P0b: Verifying environment"
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

if [[ "$SKIP_PREPROCESS" != "1" ]]; then
  section "P2: Preprocessing (MLDP + receiver features)"
  PRE_ARGS=(--config "$CONFIG")
  [[ -n "$PREPROCESS_LIMIT" ]] && PRE_ARGS+=(--limit "$PREPROCESS_LIMIT")
  "$PYTHON" -m src.preprocessing.preprocess_apks "${PRE_ARGS[@]}"
else
  echo "(Skipping preprocessing; SKIP_PREPROCESS=1)"
fi

PROCESSED_DIR="$ROOT/artifacts/processed"
if [[ ! -d "$PROCESSED_DIR" ]] || [[ -z "$(ls -A "$PROCESSED_DIR"/*.pt 2>/dev/null || true)" ]]; then
  echo "ERROR: No preprocessed feature shards under $PROCESSED_DIR"
  exit 1
fi

"$PYTHON" "$ARCHIVE_PY" export-corpus-stats \
  --profile "$PROFILE" \
  --root "$ROOT" \
  ${BMH_RUN_ID:+--run-id "$BMH_RUN_ID"}

if [[ "$SKIP_VERIFY_DATALOADER" != "1" ]]; then
  section "P3: Verifying DataLoaders and split"
  "$PYTHON" "$ROOT/scripts/verify_dataloader.py" --config "$CONFIG"
fi

if [[ "$SKIP_VERIFY_MODEL" != "1" ]]; then
  section "P4: Verifying model architecture"
  "$PYTHON" "$ROOT/scripts/verify_model.py"
fi

CHECKPOINT="$ROOT/artifacts/checkpoints/best.pt"
if [[ "$SKIP_TRAIN" != "1" ]]; then
  section "P5: Training deployment MLP"
  if [[ "$FRESH_TRAIN" == "1" && -f "$CHECKPOINT" ]]; then
    rm -f "$CHECKPOINT"
    echo "FRESH_TRAIN=1 → removed $CHECKPOINT"
  fi
  TRAIN_ARGS=(--config "$CONFIG")
  [[ -n "$EPOCHS" ]] && TRAIN_ARGS+=(--epochs "$EPOCHS")
  if [[ "$QUICK" == "1" ]]; then
    TRAIN_ARGS+=(--epochs 8 --skip-baselines)
    echo "QUICK=1 → 8 epochs, skipping sklearn baselines"
  fi
  "$PYTHON" -m src.training.train "${TRAIN_ARGS[@]}"
else
  echo "(Skipping training; SKIP_TRAIN=1)"
fi

if [[ ! -f "$CHECKPOINT" ]]; then
  echo "ERROR: Checkpoint not found: $CHECKPOINT"
  exit 1
fi

if [[ "$SKIP_EVAL" != "1" ]]; then
  section "P6: Evaluation (test split)"
  "$PYTHON" -m src.training.evaluate \
    --config "$CONFIG" \
    --metrics-out "$ROOT/artifacts/metrics/test_results.json"
else
  echo "(Skipping evaluation; SKIP_EVAL=1)"
fi

if [[ "$SKIP_EXPORT_ONNX" != "1" ]]; then
  section "P7: ONNX export"
  "$PYTHON" "$ROOT/scripts/export_onnx.py" \
    --config "$CONFIG" \
    --checkpoint "$CHECKPOINT"
else
  echo "(Skipping ONNX export; SKIP_EXPORT_ONNX=1)"
fi

if [[ "$SKIP_PARITY" != "1" ]]; then
  section "P8: ONNX parity check"
  "$PYTHON" -m src.training.parity_onnx \
    --config "$CONFIG" \
    --checkpoint "$CHECKPOINT"
else
  echo "(Skipping parity; SKIP_PARITY=1)"
fi

if [[ "$BMH_ARCHIVE" == "1" ]]; then
  if [[ "$SKIP_PLOTS" != "1" ]]; then
    section "Thesis figures"
    "$PYTHON" "$ARCHIVE_PY" plot \
      --profile "$PROFILE" \
      --root "$ROOT" \
      --run-id "$BMH_RUN_ID"
  fi
  section "Finalize run archive"
  "$ROOT/scripts/archive_run.sh" "$BMH_RUN_ID"
  section "Thesis snippet"
  "$PYTHON" "$ARCHIVE_PY" snippet \
    --profile "$PROFILE" \
    --root "$ROOT" \
    --run-id "$BMH_RUN_ID"
fi

section "Broadcast + MLDP Hybrid pipeline finished"
echo "Checkpoint:  $CHECKPOINT"
echo "Metrics:     $ROOT/artifacts/metrics/"
echo "ONNX bundle: $ROOT/artifacts/export/broadcast_mldp_hybrid/"
if [[ "$BMH_ARCHIVE" == "1" ]]; then
  echo "Run archive: $ROOT/output_archives/$BMH_RUN_ID"
  echo "  THESIS_SNIPPET: $ROOT/output_archives/$BMH_RUN_ID/THESIS_SNIPPET.md"
fi
echo "Done."
