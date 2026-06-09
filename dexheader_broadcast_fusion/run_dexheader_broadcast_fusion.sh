#!/usr/bin/env bash
# Dex header + broadcast receiver fusion — P0–P8 runner
# + figures, archive finalize, THESIS_SNIPPET (default on; DBF_ARCHIVE=0 or SKIP_ARCHIVE=1 to disable)
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$SCRIPT_DIR"
REPO_ROOT="$(cd "$ROOT/.." && pwd)"
cd "$ROOT"

PY="$("$SCRIPT_DIR/scripts/_python.sh")"
# shellcheck source=/dev/null
source "$REPO_ROOT/scripts/thesis_pythonpath.sh"
export PYTHONPATH="${SCRIPT_DIR}${PYTHONPATH:+:$PYTHONPATH}"

ARCHIVE_PY="$REPO_ROOT/scripts/thesis_run_archive.py"
PROFILE="dexheader_broadcast_fusion"
CONFIG="${CONFIG:-$ROOT/config/default.yaml}"

VERIFY_SETUP="${VERIFY_SETUP:-1}"
SKIP_INDEX="${SKIP_INDEX:-0}"
SKIP_PREPROCESS="${SKIP_PREPROCESS:-0}"
SKIP_TRAIN="${SKIP_TRAIN:-0}"
SKIP_EVAL="${SKIP_EVAL:-0}"
SKIP_EXPORT="${SKIP_EXPORT:-0}"
SKIP_PARITY="${SKIP_PARITY:-0}"
SKIP_ARCHIVE="${SKIP_ARCHIVE:-0}"
SKIP_PLOTS="${SKIP_PLOTS:-0}"
STAGE_ANDROID="${STAGE_ANDROID:-1}"
SMOKE="${SMOKE:-1}"
EPOCHS="${EPOCHS:-}"
PREPROCESS_LIMIT="${PREPROCESS_LIMIT:-}"

DBF_ARCHIVE="${DBF_ARCHIVE:-1}"
DBF_RUN_ID="${DBF_RUN_ID:-}"
if [[ "$SKIP_ARCHIVE" == "1" ]]; then
  DBF_ARCHIVE=0
fi

if [[ "$DBF_ARCHIVE" == "1" ]]; then
  if [[ -z "$DBF_RUN_ID" ]]; then
    DBF_RUN_ID="run_$(date +%Y%m%d_%H%M%S)_dbf"
  fi
  export DBF_RUN_ID
  ARCHIVE_DIR="$ROOT/output_archives/$DBF_RUN_ID"
  "$PY" "$ARCHIVE_PY" bootstrap \
    --profile "$PROFILE" \
    --root "$ROOT" \
    --run-id "$DBF_RUN_ID" \
    --config "$CONFIG"
  exec > >(tee -a "$ARCHIVE_DIR/logs/pipeline_full.log") 2>&1
fi

section() {
  echo ""
  echo "============================================================================="
  echo "  $1"
  echo "============================================================================="
}

section "Dex header + broadcast fusion — configuration"
echo "ROOT:            $ROOT"
echo "SKIP_ARCHIVE:    $SKIP_ARCHIVE"
echo "SKIP_PLOTS:      $SKIP_PLOTS"
echo "DBF_ARCHIVE:     $DBF_ARCHIVE"
echo "DBF_RUN_ID:      ${DBF_RUN_ID:-<auto when DBF_ARCHIVE=1>}"
echo "STAGE_ANDROID:   $STAGE_ANDROID"

if [[ "$VERIFY_SETUP" == "1" ]]; then
  "$PY" scripts/verify_setup.py
fi

if [[ "$SKIP_INDEX" != "1" ]]; then
  "$PY" scripts/index_dataset.py
fi

if [[ "$SKIP_PREPROCESS" != "1" ]]; then
  if [[ -n "$PREPROCESS_LIMIT" ]]; then
    "$PY" -m src.preprocessing.preprocess_apks --limit "$PREPROCESS_LIMIT"
  else
    "$PY" -m src.preprocessing.preprocess_apks
  fi
fi

INDEX_FILE="$ROOT/artifacts/dataset_index.csv"
if [[ -f "$INDEX_FILE" ]]; then
  "$PY" "$ARCHIVE_PY" export-corpus-stats \
    --profile "$PROFILE" \
    --root "$ROOT" \
    ${DBF_RUN_ID:+--run-id "$DBF_RUN_ID"}
fi

if [[ "$SKIP_TRAIN" != "1" ]]; then
  export SMOKE
  TRAIN_ARGS=()
  if [[ -n "$EPOCHS" ]]; then
    TRAIN_ARGS+=(--epochs "$EPOCHS")
  elif [[ "$SMOKE" == "1" ]]; then
    TRAIN_ARGS+=(--epochs 2)
  fi
  "$PY" -m src.training.train "${TRAIN_ARGS[@]}"
fi

if [[ "$SKIP_EVAL" != "1" ]]; then
  "$PY" -m src.training.evaluate
fi

if [[ "$SKIP_EXPORT" != "1" ]]; then
  "$PY" scripts/export_onnx.py
fi

if [[ "$SKIP_PARITY" != "1" ]]; then
  "$PY" -m src.training.parity_onnx
fi

if [[ "$DBF_ARCHIVE" == "1" ]]; then
  if [[ "$SKIP_PLOTS" != "1" ]]; then
    section "Thesis figures"
    "$PY" "$ARCHIVE_PY" plot \
      --profile "$PROFILE" \
      --root "$ROOT" \
      --run-id "$DBF_RUN_ID"
  else
    echo "(Skipping figures; SKIP_PLOTS=1)"
  fi
  section "Finalize run archive"
  "$ROOT/scripts/archive_run.sh" "$DBF_RUN_ID"
  section "Thesis snippet"
  "$PY" "$ARCHIVE_PY" snippet \
    --profile "$PROFILE" \
    --root "$ROOT" \
    --run-id "$DBF_RUN_ID"
fi

if [[ "$STAGE_ANDROID" != "0" ]]; then
  echo ""
  echo "=== Stage Android assets (P7 → vigidroid/) ==="
  bash "$REPO_ROOT/Android_Works/stage_dexheader_broadcast_fusion.sh"
fi

section "dexheader_broadcast_fusion pipeline complete"
if [[ "$DBF_ARCHIVE" == "1" ]]; then
  echo "Run archive: $ROOT/output_archives/$DBF_RUN_ID"
  echo "  THESIS_SNIPPET: $ROOT/output_archives/$DBF_RUN_ID/THESIS_SNIPPET.md"
fi
if [[ "$STAGE_ANDROID" != "0" ]]; then
  echo "Android assets: $REPO_ROOT/vigidroid/app/src/main/assets/models/dexheader_broadcast_fusion/"
fi
