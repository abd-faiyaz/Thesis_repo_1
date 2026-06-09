#!/usr/bin/env bash
# Evaluate legacy ByteCNN and XGBoost on shared thesis val/test splits.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY="${ROOT}/thesis_venv/bin/python"
if [[ ! -x "$PY" ]]; then
  PY=python3
fi

SPLIT="${SPLIT:-test}"
LIMIT_ARGS=()
if [[ -n "${LIMIT:-}" ]]; then
  LIMIT_ARGS=(--limit "$LIMIT")
fi

echo "=== Legacy ByteCNN (${SPLIT}) ==="
"$PY" "$ROOT/legacy_models/evaluate.py" --model bytecnn --split "$SPLIT" "${LIMIT_ARGS[@]}"

echo "=== Legacy XGBoost (${SPLIT}) ==="
"$PY" "$ROOT/legacy_models/evaluate.py" --model manifest_xgb --split "$SPLIT" "${LIMIT_ARGS[@]}"

echo "Done."
