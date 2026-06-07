#!/usr/bin/env bash
# Stage broadcast_mldp_hybrid ONNX export bundle into vigidroid assets (P7 → A1–A4).
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SRC="$REPO_ROOT/broadcast_mldp_hybrid/artifacts/export/broadcast_mldp_hybrid"
DEST="$REPO_ROOT/vigidroid/app/src/main/assets/models/broadcast_mldp_hybrid"
PYTHON="${REPO_ROOT}/thesis_venv/bin/python"

if [[ ! -f "$SRC/model.onnx" || ! -f "$SRC/export_manifest.json" ]]; then
  echo "ERROR: incomplete export at $SRC — run broadcast_mldp_hybrid P7 first."
  exit 1
fi

echo "Staging broadcast_mldp_hybrid"
echo "  from $SRC"
echo "  to   $DEST"

rm -rf "$DEST"
mkdir -p "$DEST/features" "$DEST/parity_samples"

cp "$SRC/model.onnx" "$SRC/export_manifest.json" "$SRC/thresholds.json" "$DEST/"
cp -r "$SRC/features/"* "$DEST/features/"
cp -r "$SRC/parity_samples/"* "$DEST/parity_samples/"

"$PYTHON" - <<PY
import json
from pathlib import Path
import numpy as np

p = Path("${DEST}/parity_samples")
d = np.load(p / "sample_vectors.npz")
out = {"vectors": np.asarray(d["vectors"], dtype=np.float32).tolist()}
out["expected_malware_probability"] = np.asarray(d["expected_malware_probability"]).tolist()
if "labels" in d:
    out["labels"] = np.asarray(d["labels"]).tolist()
if "sample_ids" in d:
    out["sample_ids"] = [str(x) for x in d["sample_ids"].tolist()]
else:
    out["sample_ids"] = [str(int(x)) for x in d["indices"].tolist()]
(p / "parity_vectors.json").write_text(json.dumps(out) + "\n", encoding="utf-8")
print("  parity_vectors.json →", p / "parity_vectors.json")
PY

echo "Regenerating A4 androidTest manifests (requires APK corpus on disk)..."
"$REPO_ROOT/broadcast_mldp_hybrid/scripts/generate_a4_parity_fixtures.sh" || {
  echo "WARN: A4 fixture generation skipped (APK corpus unavailable). Existing androidTest assets kept."
}

echo "Done."
