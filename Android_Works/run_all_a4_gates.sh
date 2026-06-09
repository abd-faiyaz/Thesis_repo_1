#!/usr/bin/env bash
# Run every Android A4 parity gate (requires connected device).
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

if ! command -v adb >/dev/null 2>&1 || ! adb devices | awk 'NR>1 && $2=="device"{found=1} END{exit !found}'; then
  echo "ERROR: no adb device connected"
  exit 1
fi

GATES=(
  run_broadcast_mldp_hybrid_a4.sh
  run_mldp_dexheader_a4.sh
  run_dexheader_broadcast_fusion_a4.sh
)

for gate in "${GATES[@]}"; do
  echo ""
  echo "############################################"
  echo "# $gate"
  echo "############################################"
  bash "$REPO_ROOT/Android_Works/$gate"
done

echo ""
echo "All A4 gates PASSED (${#GATES[@]}/${#GATES[@]})"
