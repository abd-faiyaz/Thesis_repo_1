#!/usr/bin/env bash
# Phase 1 — run PC offline test eval for all registry models, then collect + CSV.
#
# Usage:
#   ./run_offline_plotting_eval.sh              # collect from existing artifacts only
#   ./run_offline_plotting_eval.sh --run-eval   # re-run all P6 evaluate commands first
#   ./run_offline_plotting_eval.sh --validate   # registry + source path check only
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
PY="${REPO_ROOT}/thesis_venv/bin/python"
if [[ ! -x "$PY" ]]; then
  PY=python3
fi

RUN_EVAL=0
VALIDATE_ONLY=0
for arg in "$@"; do
  case "$arg" in
    --run-eval) RUN_EVAL=1 ;;
    --validate) VALIDATE_ONLY=1 ;;
    -h|--help)
      sed -n '2,8p' "$0"
      exit 0
      ;;
    *)
      echo "Unknown argument: $arg" >&2
      exit 2
      ;;
  esac
done

cd "$REPO_ROOT"

if [[ "$VALIDATE_ONLY" == "1" ]]; then
  exec "$PY" "$SCRIPT_DIR/collect_offline_test_metrics.py" --validate-only
fi

run_py() {
  local cwd="$1"
  shift
  echo "=== $* (cwd=$cwd) ==="
  (cd "$REPO_ROOT/$cwd" && PYTHONPATH="$REPO_ROOT/$cwd${PYTHONPATH:+:$PYTHONPATH}" "$PY" "$@")
}

if [[ "$RUN_EVAL" == "1" ]]; then
  echo "Running offline test evaluation for all registry models..."
  bash "$REPO_ROOT/legacy_models/run_thesis_eval.sh"

  run_py Dex_header_paper_implementation/only_base1_model -m src.training.evaluate --split test
  run_py Dex_header_paper_implementation/custom_approach/full_combined_pipeline_approach \
    -m src.training.evaluate --split test
  run_py Dex_header_paper_implementation/custom_approach/dual_branch_merge_approach \
    -m src.training.evaluate --split test
  run_py linear -m src.training.evaluate
  run_py permission_extractor -m src.training.evaluate
  run_py broadcast_mldp_hybrid -m src.training.evaluate
  run_py mldp_dexheader_cascade -m src.training.evaluate --split test
  run_py dexheader_broadcast_fusion -m src.training.evaluate

  echo "Evaluate pass complete."
else
  echo "Skipping evaluate (--run-eval not set); collecting from existing artifacts."
fi

"$PY" "$SCRIPT_DIR/collect_offline_test_metrics.py"
# Merge device columns when plot_metrics_table.json exists (do not pass --offline-only).
"$PY" "$SCRIPT_DIR/build_extended_abstract_csv.py"
echo "Done. Latest offline JSON: Shared_pipeline_Files/results/offline/latest/"
echo "CSV: Illustrations_templates/On-Device ML-Experiments - Sheet1-generated.csv"
