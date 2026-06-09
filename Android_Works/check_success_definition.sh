#!/usr/bin/env bash
# Check app_runtime_fixing.md §7 success criteria against pulled device metrics + A4 gate log.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${REPO_ROOT}/thesis_venv/bin/python"
TOOLS="$REPO_ROOT/Shared_pipeline_Files/tools"
SCAN_A_DIR="$REPO_ROOT/Shared_pipeline_Files/results/device/scan_a_all_models"
SCAN_B_DIR="$REPO_ROOT/Shared_pipeline_Files/results/device/scan_b_cascade"
MIN_SCANS="${1:-1555}"

if [[ ! -x "$PYTHON" ]]; then
  PYTHON=python3
fi

scan_a_file() {
  for f in scan_a_all_models.jsonl scan_a_all_models.json; do
    [[ -f "$SCAN_A_DIR/$f" ]] && echo "$SCAN_A_DIR/$f" && return 0
  done
  return 1
}

scan_b_file() {
  for f in scan_b_cascade.jsonl scan_b_cascade.json; do
    [[ -f "$SCAN_B_DIR/$f" ]] && echo "$SCAN_B_DIR/$f" && return 0
  done
  return 1
}

PASS=0
FAIL=0
WARN=0

check() {
  local name="$1"
  local ok="$2"
  local detail="$3"
  if [[ "$ok" == "pass" ]]; then
    echo "[PASS] $name — $detail"
    PASS=$((PASS + 1))
  elif [[ "$ok" == "warn" ]]; then
    echo "[WARN] $name — $detail"
    WARN=$((WARN + 1))
  else
    echo "[FAIL] $name — $detail"
    FAIL=$((FAIL + 1))
  fi
}

echo "=== Success definition check (app_runtime_fixing.md §7) ==="
echo ""

# Criterion 2 — A4 gates (optional log from env)
if [[ -n "${A4_GATES_PASSED:-}" ]]; then
  check "A4 parity gates on device" pass "A4_GATES_PASSED=$A4_GATES_PASSED"
else
  check "A4 parity gates on device" warn "Set A4_GATES_PASSED=1 after run_all_a4_gates.sh"
fi

# Criterion 1 & 3 — Scan A
A_FILE=""
if A_FILE="$(scan_a_file)"; then
  echo "--- Scan A: $A_FILE ---"
  if "$PYTHON" "$TOOLS/validate_scan_a.py" "$A_FILE" --min-scans "$MIN_SCANS" \
      --write-table "$REPO_ROOT/Shared_pipeline_Files/results/figures/plot_metrics_table.json"; then
    check "Scan A schema + min scans" pass "validate_scan_a OK (min=$MIN_SCANS)"
  else
    check "Scan A schema + min scans" fail "validate_scan_a failed"
  fi

  ONNX_ERR=$("$PYTHON" - <<'PY' "$A_FILE"
import json, sys
path = sys.argv[1]
onnx_models = {
    "mldp_dexheader_cascade_mode_a",
    "broadcast_mldp_hybrid",
    "dexheader_broadcast_fusion",
}
errors = []
scans = 0
with open(path, encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        rec = json.loads(line)
        if rec.get("record_type") != "scan":
            continue
        scans += 1
        for st in rec.get("stages") or []:
            if st.get("status") != "error":
                continue
            mid = st.get("model_id") or ""
            msg = st.get("error_message") or ""
            if mid in onnx_models or "@extract" in msg or "@infer" in msg or "OrtException" in msg:
                errors.append((rec.get("apk", {}).get("name"), mid, msg[:120]))
print(f"scans={scans}")
print(f"onnx_extract_errors={len(errors)}")
for e in errors[:10]:
    print("  ", e)
sys.exit(1 if errors else 0)
PY
  ) && check "Zero ONNX/extract stage errors (Scan A)" pass "$ONNX_ERR" \
    || check "Zero ONNX/extract stage errors (Scan A)" fail "$ONNX_ERR"
else
  check "Scan A metrics present" fail "missing under $SCAN_A_DIR"
fi

# Scan B
B_FILE=""
if B_FILE="$(scan_b_file)"; then
  echo ""
  echo "--- Scan B: $B_FILE ---"
  SCAN_A_ARG=()
  [[ -n "$A_FILE" ]] && SCAN_A_ARG=(--scan-a "$A_FILE")
  if "$PYTHON" "$TOOLS/validate_scan_b.py" "$B_FILE" --min-scans "$MIN_SCANS" "${SCAN_A_ARG[@]}" \
      --write-report "$SCAN_B_DIR/cascade_device_report.json"; then
    check "Scan B schema + min scans" pass "validate_scan_b OK"
  else
    check "Scan B schema + min scans" fail "validate_scan_b failed"
  fi
else
  check "Scan B metrics present" fail "missing under $SCAN_B_DIR"
fi

# Criterion 4 — UI (manual note)
check "Scan UI clarity (Phase 4)" warn "Manual: tap a row → ensemble/cascade/total_ms in detail dialog"

echo ""
echo "Summary: $PASS passed, $WARN warnings, $FAIL failed"
[[ "$FAIL" -eq 0 ]]
