#!/usr/bin/env python3
"""Build JSON parity bundles for DexheaderBroadcastFusionA4ParityTest from bundled H/R .npy."""

from __future__ import annotations

import json
import re
import struct
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
REPO = ROOT.parent
MAIN_PARITY = REPO / "vigidroid/app/src/main/assets/models/dexheader_broadcast_fusion/parity_samples"
ANDROID_TEST_PARITY = (
    REPO
    / "vigidroid/app/src/androidTest/assets/models/dexheader_broadcast_fusion/parity_samples"
)


def load_npy_f32(path: Path) -> list[float]:
    data = path.read_bytes()
    if not data.startswith(b"\x93NUMPY"):
        raise ValueError(f"Not a .npy file: {path}")
    major, minor = data[6], data[7]
    offset = 8
    if major == 1:
        header_len = struct.unpack_from("<H", data, offset)[0]
        offset += 2
    elif major in (2, 3):
        header_len = struct.unpack_from("<I", data, offset)[0]
        offset += 4
    else:
        raise ValueError(f"Unsupported npy version {major}.{minor} in {path}")
    header = data[offset : offset + header_len].decode("latin1")
    offset += header_len
    m = re.search(r"'descr':\s*'([^']+)'", header)
    if not m or m.group(1) not in ("<f4", "|f4", "float32"):
        raise ValueError(f"Expected float32 npy in {path}, header={header!r}")
    payload = data[offset:]
    count = len(payload) // 4
    return [float(x) for x in struct.unpack(f"<{count}f", payload)]


def main() -> int:
    index = json.loads((MAIN_PARITY / "index.json").read_text(encoding="utf-8"))
    headers: list[list[float]] = []
    receivers: list[list[float]] = []
    expected: list[float] = []
    sample_ids: list[str] = []
    fixtures: list[dict] = []

    for row in index:
        sid = row["dir"]
        sample_dir = MAIN_PARITY / sid
        h = load_npy_f32(sample_dir / "H.npy")
        r = load_npy_f32(sample_dir / "R.npy")
        prob = json.loads((sample_dir / "expected_prob.json").read_text(encoding="utf-8"))[
            "malware_prob"
        ]
        headers.append(h)
        receivers.append(r)
        expected.append(float(prob))
        sample_ids.append(sid)
        fixtures.append(
            {
                "sample_id": sid,
                "sha256": row.get("sha256"),
                "expected_header": h,
                "expected_receiver": r,
                "expected_malware_probability": float(prob),
            }
        )

    onnx_vectors = {
        "model_id": "dexheader_broadcast_fusion",
        "headers": headers,
        "receivers": receivers,
        "expected_malware_probability": expected,
        "sample_ids": sample_ids,
    }
    out_onnx = MAIN_PARITY / "parity_onnx_vectors.json"
    out_onnx.write_text(json.dumps(onnx_vectors, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {out_onnx} ({len(sample_ids)} samples)")

    extraction = {
        "model_id": "dexheader_broadcast_fusion",
        "domain": "dex_header_receiver_actions",
        "tolerance": 1e-4,
        "fixtures": fixtures,
    }
    ANDROID_TEST_PARITY.mkdir(parents=True, exist_ok=True)
    out_ext = ANDROID_TEST_PARITY / "parity_extraction_fixtures.json"
    out_ext.write_text(json.dumps(extraction, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {out_ext}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
