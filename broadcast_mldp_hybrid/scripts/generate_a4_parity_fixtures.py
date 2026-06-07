#!/usr/bin/env python3
"""Generate androidTest A4 parity manifests + extraction fixtures from val parity APKs."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.config import load_config
from src.features.manifest_decode import decode_manifest
from src.features.receivers import filter_receiver_system_actions, load_system_actions
from src.features.vectorize import vectorize_hybrid

ANDROID_TEST_PARITY = (
    ROOT.parent
    / "vigidroid/app/src/androidTest/assets/models/broadcast_mldp_hybrid/parity_samples"
)


def main() -> int:
    cfg = load_config()
    export_dir = cfg.paths.export
    index_path = export_dir / "parity_samples" / "index.json"
    parity_path = export_dir / "parity_samples" / "parity_vectors.json"
    val_path = cfg.paths.processed / "features_val.pt"

    if not index_path.is_file():
        print(f"ERROR: missing {index_path}", file=sys.stderr)
        return 1
    if not val_path.is_file():
        print(f"ERROR: missing {val_path}", file=sys.stderr)
        return 1

    system_actions = load_system_actions(cfg.paths.system_actions_file)
    mldp_vocab = json.loads(
        (cfg.paths.processed / "mldp_permission_vocab.json").read_text(encoding="utf-8")
    )["tokens"]
    receiver_vocab = json.loads(
        (cfg.paths.processed / "receiver_action_vocab.json").read_text(encoding="utf-8")
    )["tokens"]
    val = torch.load(val_path, weights_only=False)
    index = json.loads(index_path.read_text(encoding="utf-8"))
    parity = json.loads(parity_path.read_text(encoding="utf-8"))

    manifest_dir = ANDROID_TEST_PARITY / "manifests"
    manifest_dir.mkdir(parents=True, exist_ok=True)

    fixtures: list[dict] = []
    for i, sample in enumerate(index["samples"]):
        sid = sample["sample_id"]
        idx = int(sample["index"])
        apk = Path(val["paths"][idx])
        if not apk.is_file():
            print(f"ERROR: APK not found: {apk}", file=sys.stderr)
            return 1

        parsed = decode_manifest(apk)
        filtered = filter_receiver_system_actions(parsed.receiver_actions, system_actions)
        vec = vectorize_hybrid(
            parsed.permissions,
            filtered,
            mldp_vocab=mldp_vocab,
            receiver_vocab=receiver_vocab,
        )
        expected_vec = np.asarray(parity["vectors"][i], dtype=np.float32)
        diff = float(np.max(np.abs(vec - expected_vec)))
        if diff > 0.0:
            print(f"ERROR: vector mismatch {sid} max_diff={diff}", file=sys.stderr)
            return 1

        manifest_out = manifest_dir / f"{sid}.xml"
        with manifest_out.open("wb") as out:
            subprocess.run(
                ["unzip", "-p", str(apk), "AndroidManifest.xml"],
                check=True,
                stdout=out,
            )

        fixtures.append(
            {
                "sample_id": sid,
                "val_index": idx,
                "apk_basename": apk.name,
                "permissions": list(parsed.permissions),
                "receiver_actions": filtered,
                "expected_vector": vec.astype(np.float32).tolist(),
                "expected_malware_probability": float(parity["expected_malware_probability"][i]),
                "max_diff_vs_parity_vector": diff,
            }
        )
        print(f"  {sid} OK ({manifest_out.stat().st_size} bytes)")

    payload = {
        "model_id": cfg.model_id,
        "domain": cfg.domain,
        "tolerance": 1e-4,
        "S": len(mldp_vocab),
        "R": len(receiver_vocab),
        "total": len(mldp_vocab) + len(receiver_vocab),
        "fixtures": fixtures,
    }
    out_path = ANDROID_TEST_PARITY / "parity_extraction_fixtures.json"
    out_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {out_path} ({len(fixtures)} fixtures)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
