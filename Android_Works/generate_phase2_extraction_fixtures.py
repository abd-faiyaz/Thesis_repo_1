#!/usr/bin/env python3
"""Generate Phase 2 PC extraction fixtures for androidTest (fusion golden APKs)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
ANDROID_PHASE2 = REPO / "vigidroid/app/src/androidTest/assets/phase2"
MLDP_APK_DIR = (
    REPO
    / "vigidroid/app/src/androidTest/assets/models/mldp_dexheader_cascade/parity_samples/apks"
)
MLDP_FIXTURES = (
    REPO
    / "vigidroid/app/src/androidTest/assets/models/mldp_dexheader_cascade/"
    "parity_samples/parity_extraction_fixtures.json"
)
FUSION_ROOT = REPO / "dexheader_broadcast_fusion"


def main() -> int:
    if not MLDP_FIXTURES.is_file():
        print(f"ERROR: missing {MLDP_FIXTURES}", file=sys.stderr)
        return 1

    sys.path.insert(0, str(FUSION_ROOT))
    from src.config import load_config  # type: ignore
    from src.features.manifest_decode import decode_manifest  # type: ignore
    from src.features.normalization import load_normalization_header, transform_minmax  # type: ignore
    from src.features.receivers import (  # type: ignore
        filter_receiver_system_actions,
        load_system_actions,
    )
    from src.features.vectorize import vectorize_receiver_actions  # type: ignore
    from src.preprocessing.apk_extract import extract_apk_header_extraction  # type: ignore

    cfg = load_config()
    receiver_vocab = json.loads(
        (cfg.paths.processed / "receiver_action_vocab.json").read_text(encoding="utf-8")
    )["tokens"]
    system_actions = load_system_actions(cfg.paths.system_actions_file)
    mins, maxs, _ = load_normalization_header(cfg.paths.processed / "normalization_header.json")
    dex_cfg = cfg.dex

    payload = json.loads(MLDP_FIXTURES.read_text(encoding="utf-8"))
    fixtures_out: list[dict] = []

    for fixture in payload.get("fixtures", []):
        sid = fixture["sample_id"]
        apk = MLDP_APK_DIR / f"{sid}.apk"
        if not apk.is_file():
            print(f"ERROR: missing golden APK {apk}", file=sys.stderr)
            return 1

        parsed = decode_manifest(apk)
        filtered = filter_receiver_system_actions(parsed.receiver_actions, system_actions)
        extraction = extract_apk_header_extraction(
            apk,
            mode=str(dex_cfg["mode"]),
            pattern=str(dex_cfg["dex_pattern"]),
            max_dex=int(dex_cfg["max_dex"]),
        )
        header = transform_minmax(extraction.vector.reshape(1, -1), mins, maxs).reshape(-1)
        receiver = vectorize_receiver_actions(filtered, receiver_vocab=receiver_vocab)

        fixtures_out.append(
            {
                "sample_id": sid,
                "expected_header": [float(x) for x in header.tolist()],
                "expected_receiver": [float(x) for x in receiver.tolist()],
            }
        )
        print(f"  fusion {sid} OK")

    out_path = ANDROID_PHASE2 / "fusion_extraction_fixtures.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps(
            {
                "model_id": "dexheader_broadcast_fusion",
                "tolerance": 1e-4,
                "fixtures": fixtures_out,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"Wrote {out_path} ({len(fixtures_out)} fixtures)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
