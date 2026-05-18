#!/usr/bin/env bash
# Train Pattern A CombinedNet on cached shards (Phase 5).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
export PYTHONPATH="${ROOT}${PYTHONPATH:+:$PYTHONPATH}"
python -m src.training.train "$@"
