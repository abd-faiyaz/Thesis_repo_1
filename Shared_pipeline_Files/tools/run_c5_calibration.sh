#!/usr/bin/env bash
# Phase C5 — calibrate cascade policy, stamp export manifests, offline validation.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
if [[ -x "$REPO_ROOT/thesis_venv/bin/python" ]]; then
  PYTHON="$REPO_ROOT/thesis_venv/bin/python"
else
  PYTHON="${PYTHON:-python3}"
fi

CAL="$REPO_ROOT/Shared_pipeline_Files/calibration"
POLICY_SRC="$CAL/cascade_policy.json"
POLICY_DST="$REPO_ROOT/vigidroid/app/src/main/assets/cascade_policy.json"

echo "=== C5: collect val scores ==="
"$PYTHON" "$SCRIPT_DIR/collect_calibration_val_scores.py" --workspace "$CAL"

echo
echo "=== C5: build cascade_policy.json ==="
"$PYTHON" "$SCRIPT_DIR/build_cascade_policy.py" \
  --workspace "$CAL" \
  --tier-spec "$REPO_ROOT/Shared_pipeline_Files/data/cascade_tier_spec.json" \
  --out "$POLICY_SRC"

echo
echo "=== C5: copy policy to app assets (enabled stays false) ==="
cp "$POLICY_SRC" "$POLICY_DST"

echo
echo "=== C5: stamp val_f1 / val_accuracy on export manifests ==="
"$PYTHON" "$SCRIPT_DIR/stamp_export_manifest_metrics.py"

echo
echo "=== C5: offline cascade simulation ==="
"$PYTHON" "$SCRIPT_DIR/simulate_cascade_eval.py" \
  --policy "$POLICY_SRC" \
  --workspace "$CAL" \
  --out "$CAL/cascade_simulation_report.json"

echo
echo "=== C5 complete ==="
echo "Policy:      $POLICY_DST"
echo "Simulation:  $CAL/cascade_simulation_report.json"
echo "Next: device 400-APK eval (enabled=false vs true) → compare_cascade_eval.py"
