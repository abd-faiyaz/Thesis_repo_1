#!/usr/bin/env bash
# Copy P7 export bundle into vigidroid main assets + refresh parity_onnx_vectors.json.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
SRC="$REPO_ROOT/mldp_dexheader_cascade/artifacts/export/mldp_dexheader_cascade"
DST="$REPO_ROOT/vigidroid/app/src/main/assets/models/mldp_dexheader_cascade"

if [[ ! -f "$SRC/mode_a/model.onnx" || ! -f "$SRC/mode_b/stage1_mldp.onnx" ]]; then
  echo "ERROR: incomplete export at $SRC — run P7 export first." >&2
  exit 1
fi

echo "Deploy mldp_dexheader_cascade export bundle"
echo "  from $SRC"
echo "  to   $DST"

mkdir -p "$DST"
cp -r "$SRC/mode_a" "$SRC/mode_b" "$DST/"
cp "$SRC/thresholds.json" "$DST/"
cp -r "$SRC/features" "$DST/"
cp -r "$SRC/parity_samples" "$DST/"

bash "$SCRIPT_DIR/generate_a2_parity_vectors.sh" 2>/dev/null \
  || python3 "$SCRIPT_DIR/generate_a2_parity_vectors.py"

echo "Deployed → $DST"
