#!/usr/bin/env python3
"""Augment A1 extraction fixtures with expected ONNX scores for Android A4 parity gate."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
REPO_ROOT = ROOT.parent
ANDROID_TEST_PARITY = (
    REPO_ROOT
    / "vigidroid/app/src/androidTest/assets/models/mldp_dexheader_cascade/parity_samples"
)
FIXTURES_PATH = ANDROID_TEST_PARITY / "parity_extraction_fixtures.json"


def main() -> int:
    export_index = ROOT / "artifacts/export/mldp_dexheader_cascade/parity_samples/index.json"
    if not export_index.is_file():
        print(f"ERROR: missing {export_index}", file=sys.stderr)
        return 1
    if not FIXTURES_PATH.is_file():
        print(
            f"ERROR: missing {FIXTURES_PATH} — run generate_a1_parity_fixtures.sh first",
            file=sys.stderr,
        )
        return 1

    index = json.loads(export_index.read_text(encoding="utf-8"))
    by_id = {row["sample_id"]: row for row in index["samples"]}
    payload = json.loads(FIXTURES_PATH.read_text(encoding="utf-8"))
    fixtures = payload.get("fixtures", [])
    if not fixtures:
        print("ERROR: no fixtures in parity_extraction_fixtures.json", file=sys.stderr)
        return 1

    for fixture in fixtures:
        sid = fixture["sample_id"]
        if sid not in by_id:
            print(f"ERROR: sample {sid} missing from export index", file=sys.stderr)
            return 1
        row = by_id[sid]
        fixture["expected_mode_a_malware_prob"] = float(row["mode_a_malware_prob"])
        fixture["expected_stage1_prob"] = float(row["stage1_prob"])
        fixture["expected_stage2_prob"] = float(row["stage2_prob"])
        print(f"  {sid} scores OK")

    payload["tolerance"] = 1e-4
    FIXTURES_PATH.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    jvm_path = (
        REPO_ROOT / "vigidroid/app/src/test/resources/mldp_dexheader_cascade_a1_fixtures.json"
    )
    if jvm_path.is_file():
        jvm_payload = json.loads(jvm_path.read_text(encoding="utf-8"))
        for fixture in jvm_payload.get("fixtures", []):
            sid = fixture["sample_id"]
            if sid in by_id:
                row = by_id[sid]
                fixture["expected_mode_a_malware_prob"] = float(row["mode_a_malware_prob"])
                fixture["expected_stage1_prob"] = float(row["stage1_prob"])
                fixture["expected_stage2_prob"] = float(row["stage2_prob"])
        jvm_payload["tolerance"] = 1e-4
        jvm_path.write_text(json.dumps(jvm_payload, indent=2) + "\n", encoding="utf-8")
        print(f"Updated {jvm_path}")

    print(f"Wrote {FIXTURES_PATH} ({len(fixtures)} fixtures with expected_prob)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
