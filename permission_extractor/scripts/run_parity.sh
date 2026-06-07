#!/usr/bin/env bash
# P8 — PyTorch vs ONNX parity check.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
# shellcheck source=/dev/null
source "$ROOT/scripts/activate_thesis_env.sh"
export PYTHONPATH="${ROOT}${PYTHONPATH:+:$PYTHONPATH}"

CHECKPOINT="${CHECKPOINT:-$ROOT/artifacts/checkpoints/mldp_pruned.pth}"
CONFIG="${CONFIG:-$ROOT/config/default.yaml}"

ARGS=(--config "$CONFIG" --checkpoint "$CHECKPOINT")
"$PYTHON" "$ROOT/scripts/parity_check.py" "${ARGS[@]}" "$@"
