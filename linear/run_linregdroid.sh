#!/usr/bin/env bash
# LinRegDroid end-to-end runner (P0–P8)
# Dataset: https://huggingface.co/buckets/sakhawat2088/android-apks
# Usage:
#   APK_ROOT=/path/to/android-apks ./run_linregdroid.sh
#   SKIP_PREPROCESS=1 ./run_linregdroid.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$SCRIPT_DIR"
cd "$ROOT"

# shellcheck source=/dev/null
source "$ROOT/scripts/activate_thesis_env.sh"

APK_ROOT="${APK_ROOT:-$ROOT/data/apks}"
CONFIG="${CONFIG:-$ROOT/config/default.yaml}"
INSTALL_DEPS="${INSTALL_DEPS:-0}"
SKIP_PREPROCESS="${SKIP_PREPROCESS:-0}"
SKIP_TRAIN="${SKIP_TRAIN:-0}"
SKIP_EVAL="${SKIP_EVAL:-0}"
SKIP_EXPORT="${SKIP_EXPORT:-0}"
SKIP_PARITY="${SKIP_PARITY:-0}"
EVAL_TEMPORAL="${EVAL_TEMPORAL:-0}"

echo "LinRegDroid pipeline"
echo "  APK_ROOT=$APK_ROOT"
echo "  CONFIG=$CONFIG"

if [[ "$INSTALL_DEPS" == "1" ]]; then
  pip install -r requirements.txt
fi

echo "=== P0 verify ==="
python scripts/verify_setup.py

if [[ "$SKIP_PREPROCESS" != "1" ]]; then
  echo "=== P1–P2 preprocess ==="
  APK_ROOT="$APK_ROOT" python -m src.preprocessing.scan_dataset --config "$CONFIG" --apk-root "$APK_ROOT"
  python -m src.preprocessing.build_permission_vocab --config "$CONFIG"
  python -m src.preprocessing.extract_to_cache --config "$CONFIG"
fi

echo "=== P3 dataloader ==="
python scripts/verify_dataloader.py

if [[ "$SKIP_TRAIN" != "1" ]]; then
  echo "=== P5 train ==="
  python -m src.training.train --config "$CONFIG"
fi

if [[ "$SKIP_EVAL" != "1" ]]; then
  echo "=== P6 evaluate (val + dev_test) ==="
  EVAL_ARGS=(--config "$CONFIG" --splits val dev_test)
  if [[ "$EVAL_TEMPORAL" == "1" ]]; then
    EVAL_ARGS+=(--splits val dev_test temporal_holdout)
  fi
  python -m src.training.evaluate "${EVAL_ARGS[@]}"
fi

if [[ "$SKIP_EXPORT" != "1" ]]; then
  echo "=== P7 export ==="
  python scripts/export_onnx.py --config "$CONFIG"
fi

if [[ "$SKIP_PARITY" != "1" ]]; then
  echo "=== P8 parity ==="
  python scripts/parity_check.py --config "$CONFIG"
fi

echo "Done. Export bundle: artifacts/export/linregdroid_permission/"
echo "Copy to VigiDroid when ready (vigidroid codebase unchanged for now)."
