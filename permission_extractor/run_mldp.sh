#!/usr/bin/env bash
# MLDP end-to-end runner (P0–P8)
# Dataset: https://huggingface.co/buckets/sakhawat2088/android-apks
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"
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

echo "MLDP pipeline  APK_ROOT=$APK_ROOT"

if [[ "$INSTALL_DEPS" == "1" ]]; then
  pip install -r requirements.txt
fi

python3 scripts/verify_setup.py

if [[ "$SKIP_PREPROCESS" != "1" ]]; then
  python3 -m src.preprocessing.scan_dataset --config "$CONFIG" --apk-root "$APK_ROOT"
  python3 -m src.preprocessing.build_transactions --config "$CONFIG"
  python3 -m src.preprocessing.run_mldp_selection --config "$CONFIG"
  python3 -m src.preprocessing.extract_pruned_vectors --config "$CONFIG"
fi

python3 scripts/verify_dataloader.py

if [[ "$SKIP_TRAIN" != "1" ]]; then
  python3 -m src.training.train --config "$CONFIG"
fi

if [[ "$SKIP_EVAL" != "1" ]]; then
  EVAL_ARGS=(--config "$CONFIG" --splits val dev_test)
  if [[ "$EVAL_TEMPORAL" == "1" ]]; then
    EVAL_ARGS=(--config "$CONFIG" --splits val dev_test temporal_holdout)
  fi
  python3 -m src.training.evaluate "${EVAL_ARGS[@]}"
fi

if [[ "$SKIP_EXPORT" != "1" ]]; then
  python3 scripts/export_onnx.py --config "$CONFIG"
fi

if [[ "$SKIP_PARITY" != "1" ]]; then
  python3 scripts/parity_check.py --config "$CONFIG"
fi

echo "Done → artifacts/export/mldp_pruned_permission/"
