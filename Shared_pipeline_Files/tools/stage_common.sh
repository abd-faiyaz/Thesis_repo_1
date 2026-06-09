#!/usr/bin/env bash
# Shared helpers for staging P7 ONNX export bundles into vigidroid assets.
# Source from Android_Works/stage_*.sh — do not execute directly.

stage_common_init() {
  if [[ -n "${STAGE_COMMON_INIT:-}" ]]; then
    return 0
  fi
  STAGE_COMMON_INIT=1
  STAGE_COMMON_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
  STAGE_REPO_ROOT="$(cd "$STAGE_COMMON_DIR/../.." && pwd)"
  STAGE_VIGIDROID_MODELS="$STAGE_REPO_ROOT/vigidroid/app/src/main/assets/models"
  if [[ -x "$STAGE_REPO_ROOT/thesis_venv/bin/python" ]]; then
    STAGE_PYTHON="$STAGE_REPO_ROOT/thesis_venv/bin/python"
  else
    STAGE_PYTHON="${PYTHON:-python3}"
  fi
}

stage_write_parity_vectors_json() {
  local parity_dir="$1"
  "$STAGE_PYTHON" "$STAGE_COMMON_DIR/write_parity_vectors_json.py" "$parity_dir"
}

# Stage a single-input or dual-input ONNX bundle (model.onnx at bundle root).
stage_standard_onnx_bundle() {
  local model_id="$1"
  local src="$2"
  local dest="$STAGE_VIGIDROID_MODELS/$model_id"

  if [[ ! -f "$src/model.onnx" || ! -f "$src/export_manifest.json" ]]; then
    echo "ERROR: incomplete export bundle at $src — run P7 export first." >&2
    return 1
  fi
  if [[ ! -f "$src/thresholds.json" ]]; then
    echo "ERROR: missing thresholds.json in $src" >&2
    return 1
  fi

  echo "Staging $model_id"
  echo "  from $src"
  echo "  to   $dest"

  rm -rf "$dest"
  mkdir -p "$dest/features" "$dest/parity_samples"

  cp "$src/model.onnx" "$src/export_manifest.json" "$src/thresholds.json" "$dest/"
  cp -r "$src/features/"* "$dest/features/"

  if [[ -f "$src/parity_samples/sample_vectors.npz" ]]; then
    cp "$src/parity_samples/sample_vectors.npz" "$dest/parity_samples/"
    if [[ -f "$src/parity_samples/index.json" ]]; then
      cp "$src/parity_samples/index.json" "$dest/parity_samples/"
    fi
    stage_write_parity_vectors_json "$dest/parity_samples"
  elif [[ -d "$src/parity_samples" ]]; then
    cp -r "$src/parity_samples/"* "$dest/parity_samples/"
  else
    echo "WARN: no parity_samples under $src"
  fi

  if [[ -f "$dest/export_manifest.json" ]]; then
    "$STAGE_PYTHON" "$STAGE_COMMON_DIR/stamp_export_manifest_metrics.py" \
      --model-id "$model_id" 2>/dev/null || true
  fi
}

# Stage a full export tree (e.g. dexheader_broadcast_fusion per-sample parity dirs).
stage_copytree_bundle() {
  local model_id="$1"
  local src="$2"
  local dest="$STAGE_VIGIDROID_MODELS/$model_id"

  if [[ ! -f "$src/model.onnx" || ! -f "$src/export_manifest.json" ]]; then
    echo "ERROR: incomplete export bundle at $src — run P7 export first." >&2
    return 1
  fi

  echo "Staging $model_id (full bundle copy)"
  echo "  from $src"
  echo "  to   $dest"

  rm -rf "$dest"
  mkdir -p "$dest"
  cp -r "$src/"* "$dest/"
}
