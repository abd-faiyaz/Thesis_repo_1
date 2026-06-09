#!/usr/bin/env bash
# Generate all supervisor thesis figures from plot_metrics_table.json.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
PYTHON="${REPO_ROOT}/thesis_venv/bin/python"
if [[ ! -x "$PYTHON" ]]; then
  PYTHON="python3"
fi
TABLE="${1:-$REPO_ROOT/Shared_pipeline_Files/results/figures/plot_metrics_table.json}"
OUT="${2:-$REPO_ROOT/Shared_pipeline_Files/results/figures/templates}"

if [[ ! -f "$TABLE" ]]; then
  echo "Missing plot table: $TABLE (run aggregate_plot_metrics.py first)" >&2
  exit 1
fi

mkdir -p "$OUT"
COMMON=(--table "$TABLE" --out-dir "$OUT")

PLOTS=(
  plot_apk_size_vs_detection_time.py
  plot_inference_breakdown_stacked.py
  plot_inference_time_vs_apk_size.py
  plot_accuracy_vs_ram.py
  plot_accuracy_vs_latency.py
  plot_model_vs_resources.py
  plot_performance_tradeoff.py
  plot_cascade_exit_tiers.py
)

EXTENDED_OUT="$REPO_ROOT/extended_abstract/plots_and_table/Generated"

for script in "${PLOTS[@]}"; do
  echo "=== $script ==="
  "$PYTHON" "$SCRIPT_DIR/$script" "${COMMON[@]}"
done

"$PYTHON" "$SCRIPT_DIR/plot_inference_time_vs_apk_size.py" --table "$TABLE" --out-dir "$EXTENDED_OUT"
"$PYTHON" "$SCRIPT_DIR/plot_model_vs_resources.py" --table "$TABLE" --out-dir "$EXTENDED_OUT"

echo "Done. Figures in $OUT"
echo "Extended-abstract copies in $EXTENDED_OUT"
