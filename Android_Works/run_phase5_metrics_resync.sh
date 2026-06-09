#!/usr/bin/env bash
# Phase 5 — metrics & thesis pipeline re-sync after app runtime fixes (P0–P4).
#
# Offline (no phone):
#   ./run_phase5_metrics_resync.sh --offline-only
#
# Full re-sync (phone connected, scans already run on device):
#   ./run_phase5_metrics_resync.sh
#
# Options:
#   --offline-only     Skip adb pull; run calibration + offline plots only
#   --skip-manifest    Skip build_device_eval_manifest.py
#   --skip-calibration Skip collect_calibration_val_scores.py
#   --skip-plots       Skip run_e2e_plotting_pipeline.sh
#   --run-eval         Re-run PC offline test eval before plotting
#   --allow-missing    Pass --allow-missing to calibration collector
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
TOOLS="$REPO_ROOT/Shared_pipeline_Files/tools"
PYTHON="${REPO_ROOT}/thesis_venv/bin/python"
if [[ ! -x "$PYTHON" ]]; then
  PYTHON=python3
fi

OFFLINE_ONLY=0
SKIP_MANIFEST=0
SKIP_CALIBRATION=0
SKIP_PLOTS=0
RUN_EVAL=0
ALLOW_MISSING=0
MIN_SCANS=1

while [[ $# -gt 0 ]]; do
  case "$1" in
    --offline-only) OFFLINE_ONLY=1 ;;
    --skip-manifest) SKIP_MANIFEST=1 ;;
    --skip-calibration) SKIP_CALIBRATION=1 ;;
    --skip-plots) SKIP_PLOTS=1 ;;
    --run-eval) RUN_EVAL=1 ;;
    --allow-missing) ALLOW_MISSING=1 ;;
    --min-scans) MIN_SCANS="${2:?}"; shift ;;
    -h|--help)
      sed -n '2,18p' "$0"
      exit 0
      ;;
    *)
      echo "Unknown option: $1" >&2
      exit 2
      ;;
  esac
  shift
done

echo "=== Phase 5 — metrics & thesis pipeline re-sync ==="
echo "Repo: $REPO_ROOT"

if [[ "$SKIP_MANIFEST" -eq 0 ]]; then
  if [[ -d /mnt/Files/thesis_full_dataset ]]; then
    echo "=== 5.1 Device eval manifest ==="
    "$PYTHON" "$SCRIPT_DIR/build_device_eval_manifest.py"
  else
    echo "Skip 5.1 — /mnt/Files/thesis_full_dataset not mounted"
  fi
fi

if [[ "$SKIP_CALIBRATION" -eq 0 ]]; then
  echo "=== 5.4 Collect calibration val_scores ==="
  CAL_ARGS=(--workspace "$REPO_ROOT/Shared_pipeline_Files/calibration")
  if [[ "$ALLOW_MISSING" -eq 1 ]]; then
    CAL_ARGS+=(--allow-missing)
  fi
  "$PYTHON" "$TOOLS/collect_calibration_val_scores.py" "${CAL_ARGS[@]}"
fi

if [[ "$OFFLINE_ONLY" -eq 0 ]]; then
  if adb get-state >/dev/null 2>&1; then
    echo "=== 5.2 Pull Scan A (all models) ==="
    "$TOOLS/pull_device_metrics.sh" scan_a_all_models || true
    echo "=== 5.2 Pull Scan B (cascade) ==="
    "$TOOLS/pull_device_metrics.sh" scan_b_cascade || true

    SCAN_A="$REPO_ROOT/Shared_pipeline_Files/results/device/scan_a_all_models"
    for f in scan_a_all_models.jsonl scan_a_all_models.json all_scan_metrics.jsonl; do
      if [[ -f "$SCAN_A/$f" ]]; then
        echo "=== 5.3 Validate Scan A ($f) ==="
        "$PYTHON" "$TOOLS/validate_scan_a.py" "$SCAN_A/$f" \
          --min-scans "$MIN_SCANS" \
          --write-table "$REPO_ROOT/Shared_pipeline_Files/results/figures/plot_metrics_table.json" \
          || true
        break
      fi
    done

    SCAN_B="$REPO_ROOT/Shared_pipeline_Files/results/device/scan_b_cascade"
    for f in scan_b_cascade.jsonl scan_b_cascade.json all_scan_metrics.jsonl; do
      if [[ -f "$SCAN_B/$f" ]]; then
        echo "=== 5.3 Validate Scan B ($f) ==="
        "$PYTHON" "$TOOLS/validate_scan_b.py" "$SCAN_B/$f" || true
        break
      fi
    done
  else
    echo "No adb device — skip 5.2/5.3 pull+validate."
    echo "Run Scan A/B on phone first, then re-run without --offline-only."
    OFFLINE_ONLY=1
  fi
fi

if [[ "$SKIP_PLOTS" -eq 0 ]]; then
  echo "=== 5.5 Thesis plotting pipeline ==="
  PLOT_ARGS=(--allow-partial)
  if [[ "$OFFLINE_ONLY" -eq 1 ]]; then
    PLOT_ARGS+=(--skip-device)
  fi
  if [[ "$RUN_EVAL" -eq 1 ]]; then
    PLOT_ARGS+=(--run-eval)
  fi
  "$TOOLS/run_e2e_plotting_pipeline.sh" "${PLOT_ARGS[@]}"
fi

echo ""
echo "Phase 5 complete."
echo "  Calibration: $REPO_ROOT/Shared_pipeline_Files/calibration/"
echo "  Device pulls: $REPO_ROOT/Shared_pipeline_Files/results/device/"
echo "  Figures: $REPO_ROOT/Shared_pipeline_Files/results/figures/"
echo ""
echo "Device eval (manual, if not done):"
echo "  1. ./Android_Works/run_all_a4_gates.sh"
echo "  2. $TOOLS/run_phase3_device_scan_a.sh   # Ablation / all models"
echo "  3. $TOOLS/run_phase4_device_scan_b.sh   # Cascade mode"
echo "  4. Re-run: $0"
