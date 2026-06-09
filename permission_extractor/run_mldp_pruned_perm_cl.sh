#!/usr/bin/env bash
# =============================================================================
# run_mldp_pruned_perm_cl.sh
# End-to-end runner for MLDP-pruned permission classifier.
#
# Pipeline:
#   P0  verify_setup (+ optional pip install)
#   P1  scan_dataset → dataset_index.csv
#   P2  build_transactions + run_mldp_selection + extract_pruned_vectors
#   P3  verify_dataloader
#   P5  train LinearSVC / Tiny MLP (best on val F1)
#   P6  evaluate val + temporal_holdout (primary test → test_results.json)
#   P7  export ONNX bundle (artifacts/export/mldp_pruned_permission/)
#   P8  PyTorch vs ONNX parity
#   +   figures, archive finalize, THESIS_SNIPPET (default on; MLDP_ARCHIVE=0 or SKIP_ARCHIVE=1 to disable)
#   +   Android asset staging by default (set STAGE_ANDROID=0 to skip)
#
# Usage:
#   ./run_mldp_pruned_perm_cl.sh
#   APK_ROOT=/path/to/android-apks ./run_mldp_pruned_perm_cl.sh
#   SKIP_PREPROCESS=1 ./run_mldp_pruned_perm_cl.sh
#   STAGE_ANDROID=0 ./run_mldp_pruned_perm_cl.sh
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
PROFILE="mldp_pruned_permission"

APK_ROOT="${APK_ROOT:-}"
CONFIG="${CONFIG:-$ROOT/config/default.yaml}"
INSTALL_DEPS="${INSTALL_DEPS:-0}"
VERIFY_SETUP="${VERIFY_SETUP:-1}"
SKIP_PREPROCESS="${SKIP_PREPROCESS:-0}"
SKIP_VERIFY_DATALOADER="${SKIP_VERIFY_DATALOADER:-0}"
SKIP_TRAIN="${SKIP_TRAIN:-0}"
SKIP_EVAL="${SKIP_EVAL:-0}"
SKIP_EXPORT="${SKIP_EXPORT:-0}"
SKIP_PARITY="${SKIP_PARITY:-0}"
SKIP_ARCHIVE="${SKIP_ARCHIVE:-0}"
SKIP_PLOTS="${SKIP_PLOTS:-0}"
FRESH_TRAIN="${FRESH_TRAIN:-0}"
PREPROCESS_LIMIT="${PREPROCESS_LIMIT:-}"

MLDP_ARCHIVE="${MLDP_ARCHIVE:-1}"
MLDP_RUN_ID="${MLDP_RUN_ID:-}"
STAGE_ANDROID="${STAGE_ANDROID:-1}"
if [[ "$SKIP_ARCHIVE" == "1" ]]; then
  MLDP_ARCHIVE=0
fi

if [[ "$MLDP_ARCHIVE" == "1" ]]; then
  if [[ -z "$MLDP_RUN_ID" ]]; then
    MLDP_RUN_ID="run_$(date +%Y%m%d_%H%M%S)_mldp"
  fi
  export MLDP_RUN_ID
  ARCHIVE_DIR="$ROOT/output_archives/$MLDP_RUN_ID"
  "$PYTHON" "$ARCHIVE_PY" bootstrap \
    --profile "$PROFILE" \
    --root "$ROOT" \
    --run-id "$MLDP_RUN_ID" \
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

section "MLDP-pruned permission classifier — configuration"
echo "ROOT:            $ROOT"
echo "APK_ROOT:        ${APK_ROOT:-<from config>}"
echo "CONFIG:          $CONFIG"
echo "SKIP_PREPROCESS: $SKIP_PREPROCESS"
echo "SKIP_VERIFY_DL:  $SKIP_VERIFY_DATALOADER"
echo "SKIP_TRAIN:      $SKIP_TRAIN"
echo "SKIP_EVAL:       $SKIP_EVAL"
echo "SKIP_EXPORT:     $SKIP_EXPORT"
echo "SKIP_PARITY:     $SKIP_PARITY"
echo "SKIP_ARCHIVE:    $SKIP_ARCHIVE"
echo "SKIP_PLOTS:      $SKIP_PLOTS"
echo "FRESH_TRAIN:     $FRESH_TRAIN"
echo "MLDP_ARCHIVE:    $MLDP_ARCHIVE"
echo "MLDP_RUN_ID:     ${MLDP_RUN_ID:-<auto when MLDP_ARCHIVE=1>}"
echo "STAGE_ANDROID:   $STAGE_ANDROID"

if [[ "$INSTALL_DEPS" == "1" ]]; then
  section "Installing dependencies"
  REQ="$REPO_ROOT/requirements-thesis-all.txt"
  if [[ ! -f "$REQ" ]]; then
    REQ="$ROOT/requirements.txt"
  fi
  "$PYTHON" -m pip install -r "$REQ"
fi

if [[ "$VERIFY_SETUP" == "1" ]]; then
  section "P0: Verifying environment"
  "$PYTHON" "$ROOT/scripts/verify_setup.py"
fi

if [[ "$SKIP_PREPROCESS" != "1" ]]; then
  section "P1–P2: Preprocessing (MLDP selection + pruned vectors)"
  SCAN_ARGS=(--config "$CONFIG")
  [[ -n "$APK_ROOT" ]] && SCAN_ARGS+=(--apk-root "$APK_ROOT")
  [[ -n "$PREPROCESS_LIMIT" ]] && SCAN_ARGS+=(--limit "$PREPROCESS_LIMIT")
  "$PYTHON" -m src.preprocessing.scan_dataset "${SCAN_ARGS[@]}"
  "$PYTHON" -m src.preprocessing.build_transactions --config "$CONFIG"
  "$PYTHON" -m src.preprocessing.run_mldp_selection --config "$CONFIG"
  "$PYTHON" -m src.preprocessing.extract_pruned_vectors --config "$CONFIG"
else
  echo "(Skipping preprocessing; SKIP_PREPROCESS=1)"
fi

INDEX_FILE="$ROOT/artifacts/dataset_index.csv"
if [[ ! -f "$INDEX_FILE" ]]; then
  echo "ERROR: Dataset index not found: $INDEX_FILE"
  exit 1
fi

"$PYTHON" "$ARCHIVE_PY" export-corpus-stats \
  --profile "$PROFILE" \
  --root "$ROOT" \
  ${MLDP_RUN_ID:+--run-id "$MLDP_RUN_ID"}

if [[ "$SKIP_VERIFY_DATALOADER" != "1" ]]; then
  section "P3: Verifying DataLoaders and split"
  "$PYTHON" "$ROOT/scripts/verify_dataloader.py" --config "$CONFIG"
fi

CHECKPOINT="$ROOT/artifacts/checkpoints/mldp_pruned.pth"
if [[ "$SKIP_TRAIN" != "1" ]]; then
  section "P5: Training (LinearSVC / Tiny MLP selection)"
  if [[ "$FRESH_TRAIN" == "1" && -f "$CHECKPOINT" ]]; then
    rm -f "$CHECKPOINT"
    echo "FRESH_TRAIN=1 → removed $CHECKPOINT"
  fi
  "$PYTHON" -m src.training.train --config "$CONFIG"
else
  echo "(Skipping training; SKIP_TRAIN=1)"
fi

if [[ ! -f "$CHECKPOINT" ]]; then
  echo "ERROR: Checkpoint not found: $CHECKPOINT"
  exit 1
fi

if [[ "$SKIP_EVAL" != "1" ]]; then
  section "P6: Evaluation (val + temporal_holdout test)"
  "$PYTHON" -m src.training.evaluate --config "$CONFIG"
else
  echo "(Skipping evaluation; SKIP_EVAL=1)"
fi

if [[ "$SKIP_EXPORT" != "1" ]]; then
  section "P7: ONNX export"
  "$PYTHON" "$ROOT/scripts/export_onnx.py" --config "$CONFIG" --checkpoint "$CHECKPOINT"
else
  echo "(Skipping export; SKIP_EXPORT=1)"
fi

if [[ "$SKIP_PARITY" != "1" ]]; then
  section "P8: ONNX parity"
  "$PYTHON" "$ROOT/scripts/parity_check.py" --config "$CONFIG" --checkpoint "$CHECKPOINT"
else
  echo "(Skipping parity; SKIP_PARITY=1)"
fi

if [[ "$MLDP_ARCHIVE" == "1" ]]; then
  if [[ "$SKIP_PLOTS" != "1" ]]; then
    section "Thesis figures"
    "$PYTHON" "$ARCHIVE_PY" plot \
      --profile "$PROFILE" \
      --root "$ROOT" \
      --run-id "$MLDP_RUN_ID"
  fi
  section "Finalize run archive"
  "$ROOT/scripts/archive_run.sh" "$MLDP_RUN_ID"
  section "Thesis snippet"
  "$PYTHON" "$ARCHIVE_PY" snippet \
    --profile "$PROFILE" \
    --root "$ROOT" \
    --run-id "$MLDP_RUN_ID"
fi

if [[ "$STAGE_ANDROID" != "0" ]]; then
  section "Stage Android assets (P7 → vigidroid/)"
  bash "$REPO_ROOT/Android_Works/stage_mldp_pruned_permission.sh"
fi

section "MLDP pipeline finished"
echo "Checkpoint:  $CHECKPOINT"
echo "Metrics:     $ROOT/artifacts/metrics/"
echo "ONNX bundle: $ROOT/artifacts/export/mldp_pruned_permission/"
if [[ "$MLDP_ARCHIVE" == "1" ]]; then
  echo "Run archive: $ROOT/output_archives/$MLDP_RUN_ID"
  echo "  THESIS_SNIPPET: $ROOT/output_archives/$MLDP_RUN_ID/THESIS_SNIPPET.md"
fi
if [[ "$STAGE_ANDROID" != "0" ]]; then
  echo "Android assets: $REPO_ROOT/vigidroid/app/src/main/assets/models/mldp_pruned_permission/"
fi
echo "Done."
