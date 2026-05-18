#!/usr/bin/env bash
# Bundle key artifacts for copying off the remote training machine (Phase 6).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

ARTIFACTS_ROOT="${ARTIFACTS_ROOT:-$ROOT/artifacts}"
BUNDLE="${BUNDLE:-$ARTIFACTS_ROOT/pattern_a_bundle.tar.gz}"
CONFIG="${CONFIG:-$ROOT/config/default.yaml}"

echo "Packaging Pattern A artifacts → $BUNDLE"
echo "  ARTIFACTS_ROOT=$ARTIFACTS_ROOT"

REL_FILES=(
  "vocab.json"
  "normalization_header.json"
  "class_balance.json"
  "dataset_index.csv"
  "processed/manifest_train.json"
  "processed/manifest_val.json"
  "checkpoints/best.pt"
  "checkpoints/latest.pt"
  "checkpoints/metrics_val.json"
  "failed_apks.log"
  "dex_stats.json"
)

MISSING=0
TO_TAR=()
for rel in "${REL_FILES[@]}"; do
  f="$ARTIFACTS_ROOT/$rel"
  if [[ -f "$f" ]]; then
    TO_TAR+=("$f")
  else
    echo "  (skip missing) $f"
    MISSING=$((MISSING + 1))
  fi
done

if [[ -f "$CONFIG" ]]; then
  TO_TAR+=("$CONFIG")
fi

if [[ ${#TO_TAR[@]} -eq 0 ]]; then
  echo "ERROR: No files to package."
  exit 1
fi

mkdir -p "$(dirname "$BUNDLE")"
tar -czf "$BUNDLE" "${TO_TAR[@]}"
echo "Created $BUNDLE (${#TO_TAR[@]} files, $MISSING optional paths missing)"
echo ""
echo "Note: shard .npz files are NOT included (too large). Copy $ARTIFACTS_ROOT/processed/shards/ separately if needed."
