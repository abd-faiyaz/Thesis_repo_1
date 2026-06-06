#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
# shellcheck source=/dev/null
source "$ROOT/scripts/activate_thesis_env.sh"

echo "=== P1 scan ==="
python -m src.preprocessing.scan_dataset "$@"

echo "=== P2 vocab ==="
python -m src.preprocessing.build_permission_vocab "$@"

echo "=== P2 extract ==="
python -m src.preprocessing.extract_to_cache "$@"
