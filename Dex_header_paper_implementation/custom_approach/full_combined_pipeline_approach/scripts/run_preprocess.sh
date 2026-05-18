#!/usr/bin/env bash
# Full Pattern A preprocessing: index → lexicon → header norm → shard extract.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
export PYTHONPATH="${ROOT}${PYTHONPATH:+:$PYTHONPATH}"

echo "=== 1/4 scan_dataset ==="
python -m src.preprocessing.scan_dataset "$@"

echo "=== 2/4 build_lexicon (train) ==="
python -m src.preprocessing.build_lexicon "$@"

echo "=== 3/4 fit_header_norm (train) ==="
python -m src.preprocessing.fit_header_norm "$@"

echo "=== 4/4 extract_to_cache (train + val) ==="
python -m src.preprocessing.extract_to_cache "$@"

echo "Preprocessing complete."
