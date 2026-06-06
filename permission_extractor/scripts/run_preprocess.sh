#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
source "$ROOT/scripts/activate_thesis_env.sh"
python -m src.preprocessing.scan_dataset "$@"
python -m src.preprocessing.build_transactions "$@"
python -m src.preprocessing.run_mldp_selection "$@"
python -m src.preprocessing.extract_pruned_vectors "$@"
