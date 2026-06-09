#!/usr/bin/env bash
# End-to-end PC plotting pipeline (plan §1.4–1.5).
#
# Usage:
#   ./run_e2e_plotting_pipeline.sh --skip-device     # offline + aggregate partial + plots
#   ./run_e2e_plotting_pipeline.sh                   # after POCO Scan A/B pulled
#   ./run_e2e_plotting_pipeline.sh --run-eval        # re-run all offline evaluates first
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
PYTHON="${REPO_ROOT}/thesis_venv/bin/python"
if [[ ! -x "$PYTHON" ]]; then
  PYTHON="python3"
fi

SKIP_DEVICE=0
SKIP_OFFLINE=0
RUN_EVAL=0
ALLOW_PARTIAL=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --skip-device) SKIP_DEVICE=1; ALLOW_PARTIAL=1 ;;
    --skip-offline) SKIP_OFFLINE=1 ;;
    --run-eval) RUN_EVAL=1 ;;
    --allow-partial) ALLOW_PARTIAL=1 ;;
    -h|--help)
      sed -n '2,12p' "$0"
      exit 0
      ;;
    *)
      echo "Unknown option: $1" >&2
      exit 2
      ;;
  esac
  shift
done

if [[ "$SKIP_OFFLINE" -eq 0 ]]; then
  echo "=== Offline test metrics ==="
  if [[ "$RUN_EVAL" -eq 1 ]]; then
    "$SCRIPT_DIR/run_offline_plotting_eval.sh" --run-eval
  else
    "$SCRIPT_DIR/run_offline_plotting_eval.sh"
  fi
  "$PYTHON" "$SCRIPT_DIR/collect_offline_test_metrics.py"
fi

AGG_ARGS=()
if [[ "$ALLOW_PARTIAL" -eq 1 ]]; then
  AGG_ARGS+=(--allow-partial)
fi
if [[ "$SKIP_DEVICE" -eq 1 ]]; then
  AGG_ARGS+=(--offline-only)
fi

if [[ "$SKIP_DEVICE" -eq 0 ]]; then
  SCAN_A="$REPO_ROOT/Shared_pipeline_Files/results/device/scan_a_all_models"
  SCAN_B="$REPO_ROOT/Shared_pipeline_Files/results/device/scan_b_cascade"
  scan_a_metrics=""
  for f in scan_a_all_models.jsonl scan_a_all_models.json all_scan_metrics.jsonl all_scan_metrics.json; do
    if [[ -f "$SCAN_A/$f" ]]; then scan_a_metrics="$SCAN_A/$f"; break; fi
  done
  if [[ -n "$scan_a_metrics" ]]; then
    echo "=== Validate Scan A ==="
    "$PYTHON" "$SCRIPT_DIR/validate_scan_a.py" "$scan_a_metrics" || true
  else
    echo "Warning: no Scan A pull at $SCAN_A" >&2
    ALLOW_PARTIAL=1
    AGG_ARGS=(--allow-partial)
  fi
  scan_b_metrics=""
  for f in scan_b_cascade.jsonl scan_b_cascade.json all_scan_metrics.jsonl all_scan_metrics.json; do
    if [[ -f "$SCAN_B/$f" ]]; then scan_b_metrics="$SCAN_B/$f"; break; fi
  done
  if [[ -n "$scan_b_metrics" ]]; then
    echo "=== Validate Scan B ==="
    "$PYTHON" "$SCRIPT_DIR/validate_scan_b.py" "$scan_b_metrics" || true
  fi
fi

echo "=== Aggregate plot metrics ==="
"$PYTHON" "$SCRIPT_DIR/aggregate_plot_metrics.py" "${AGG_ARGS[@]}"

echo "=== Thesis figures ==="
"$SCRIPT_DIR/run_all_thesis_plots.sh"

CSV_OUT="$REPO_ROOT/Illustrations_templates/On-Device ML-Experiments - Sheet1-generated.csv"
echo "=== Extended-abstract CSV ==="
"$PYTHON" "$SCRIPT_DIR/build_extended_abstract_csv.py" --out "$CSV_OUT"

echo "=== Sufficiency report ==="
"$PYTHON" "$SCRIPT_DIR/generate_plotting_sufficiency_report.py" --csv "$CSV_OUT" || true

echo "Done. See:"
echo "  $REPO_ROOT/Shared_pipeline_Files/results/figures/plot_metrics_table.json"
echo "  $REPO_ROOT/Shared_pipeline_Files/results/figures/templates/"
echo "  $CSV_OUT"
echo "  $REPO_ROOT/Shared_pipeline_Files/results/figures/plotting_sufficiency_report.md"
