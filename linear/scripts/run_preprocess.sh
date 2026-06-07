#!/usr/bin/env bash
# P1–P2 — scan APKs, build permission vocab, extract feature cache.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
# shellcheck source=/dev/null
source "$ROOT/scripts/activate_thesis_env.sh"
export PYTHONPATH="${ROOT}${PYTHONPATH:+:$PYTHONPATH}"

CONFIG="${CONFIG:-$ROOT/config/default.yaml}"
ARGS=(--config "$CONFIG")
if [[ -n "${APK_ROOT:-}" ]]; then
  ARGS+=(--apk-root "$APK_ROOT")
fi
if [[ -n "${PREPROCESS_LIMIT:-}" ]]; then
  ARGS+=(--limit "$PREPROCESS_LIMIT")
fi

echo "=== P1 scan ==="
"$PYTHON" -m src.preprocessing.scan_dataset "${ARGS[@]}" "$@"

echo "=== P2 vocab ==="
"$PYTHON" -m src.preprocessing.build_permission_vocab --config "$CONFIG"

echo "=== P2 extract ==="
"$PYTHON" -m src.preprocessing.extract_to_cache --config "$CONFIG"
