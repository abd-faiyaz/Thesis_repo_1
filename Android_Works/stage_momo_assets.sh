#!/usr/bin/env bash
# Stage LinRegDroid + MLDP ONNX export bundles into vigidroid assets (Phase 5).
# Excludes PC-only artifacts (coefficients.json, mldp_rules.json).
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VIGIDROID_MODELS="$REPO_ROOT/vigidroid/app/src/main/assets/models"
PYTHON="${REPO_ROOT}/thesis_venv/bin/python"

write_parity_vectors_json() {
  local parity_dir="$1"
  "$PYTHON" - <<PY
import json
from pathlib import Path
import numpy as np

p = Path("${parity_dir}")
d = np.load(p / "sample_vectors.npz")
out = {"vectors": np.asarray(d["vectors"], dtype=np.float32).tolist()}
if "expected_malware_probability" in d:
    out["expected_malware_probability"] = np.asarray(d["expected_malware_probability"]).tolist()
elif "expected_scores" in d:
    out["expected_scores"] = np.asarray(d["expected_scores"]).tolist()
if "labels" in d:
    out["labels"] = np.asarray(d["labels"]).tolist()
if "sample_ids" in d:
    out["sample_ids"] = [str(x) for x in d["sample_ids"].tolist()]
elif "indices" in d:
    out["sample_ids"] = [str(int(x)) for x in d["indices"].tolist()]
else:
    out["sample_ids"] = [str(i) for i in range(len(out["vectors"]))]
(p / "parity_vectors.json").write_text(json.dumps(out) + "\n", encoding="utf-8")
print("  parity_vectors.json →", p / "parity_vectors.json")
PY
}

stage_bundle() {
  local model_id="$1"
  local src="$2"
  local dest="$VIGIDROID_MODELS/$model_id"

  if [[ ! -f "$src/model.onnx" || ! -f "$src/export_manifest.json" ]]; then
    echo "ERROR: incomplete export bundle at $src — run P7 export first."
    exit 1
  fi

  echo "Staging $model_id"
  echo "  from $src"
  echo "  to   $dest"
  rm -rf "$dest"
  mkdir -p "$dest/features" "$dest/parity_samples"

  cp "$src/model.onnx" "$src/export_manifest.json" "$src/thresholds.json" "$dest/"
  cp -r "$src/features/"* "$dest/features/"
  cp "$src/parity_samples/sample_vectors.npz" "$dest/parity_samples/"
  if [[ -f "$src/parity_samples/index.json" ]]; then
    cp "$src/parity_samples/index.json" "$dest/parity_samples/"
  fi
  write_parity_vectors_json "$dest/parity_samples"
}

stage_bundle "linregdroid_permission" \
  "$REPO_ROOT/linear/artifacts/export/linregdroid_permission"

stage_bundle "mldp_pruned_permission" \
  "$REPO_ROOT/permission_extractor/artifacts/export/mldp_pruned_permission"

echo "Done. Android assets staged under vigidroid/app/src/main/assets/models/"
