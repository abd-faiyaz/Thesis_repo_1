#!/usr/bin/env bash
# A4 — refresh extraction fixtures (expected scores) + ONNX parity vectors.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$SCRIPT_DIR/../.." && pwd)"

bash "$SCRIPT_DIR/generate_a1_parity_fixtures.sh"
bash "$SCRIPT_DIR/generate_a2_parity_vectors.sh"
python3 "$SCRIPT_DIR/generate_a4_parity_fixtures.py"
