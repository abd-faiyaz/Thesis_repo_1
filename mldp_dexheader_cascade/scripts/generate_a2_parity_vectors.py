#!/usr/bin/env python3
"""Write parity_onnx_vectors.json for Android A2/A4 ONNX inference tests."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
EXPORT = ROOT / "artifacts/export/mldp_dexheader_cascade"
OUT_MAIN = (
    ROOT.parent
    / "vigidroid/app/src/main/assets/models/mldp_dexheader_cascade/parity_samples/parity_onnx_vectors.json"
)


def main() -> int:
    index_path = EXPORT / "parity_samples" / "index.json"
    if not index_path.is_file():
        print(f"ERROR: missing {index_path}", file=sys.stderr)
        return 1

    index = json.loads(index_path.read_text(encoding="utf-8"))
    vectors: list[list[float]] = []
    x_s_vectors: list[list[float]] = []
    h_vectors: list[list[float]] = []
    mode_a_expected: list[float] = []
    stage1_expected: list[float] = []
    stage2_expected: list[float] = []
    sample_ids: list[str] = []

    for row in index["samples"]:
        sid = row["sample_id"]
        sample_dir = EXPORT / "parity_samples" / sid
        expected = json.loads((sample_dir / "expected_prob.json").read_text(encoding="utf-8"))
        vectors.append(np.load(sample_dir / "x.npy").astype(np.float32).ravel().tolist())
        x_s_vectors.append(np.load(sample_dir / "x_S.npy").astype(np.float32).ravel().tolist())
        h_vectors.append(np.load(sample_dir / "H.npy").astype(np.float32).ravel().tolist())
        mode_a_expected.append(float(expected["mode_a_malware_prob"]))
        stage1_expected.append(float(expected["stage1_prob"]))
        stage2_expected.append(float(expected["stage2_prob"]))
        sample_ids.append(sid)

    payload = {
        "model_id": "mldp_dexheader_cascade",
        "tolerance": 1e-4,
        "sample_ids": sample_ids,
        "vectors": vectors,
        "x_s_vectors": x_s_vectors,
        "h_vectors": h_vectors,
        "expected_mode_a_malware_prob": mode_a_expected,
        "expected_stage1_prob": stage1_expected,
        "expected_stage2_prob": stage2_expected,
    }
    OUT_MAIN.parent.mkdir(parents=True, exist_ok=True)
    OUT_MAIN.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {OUT_MAIN} ({len(sample_ids)} samples)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
