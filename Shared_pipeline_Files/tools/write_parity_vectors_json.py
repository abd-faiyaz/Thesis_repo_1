#!/usr/bin/env python3
"""Build parity_vectors.json from export bundle parity_samples/sample_vectors.npz."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np


def build_parity_vectors(parity_dir: Path) -> dict:
    npz_path = parity_dir / "sample_vectors.npz"
    if not npz_path.is_file():
        raise FileNotFoundError(f"Missing {npz_path}")

    data = np.load(npz_path)
    keys = set(data.files)

    def _scores_array() -> np.ndarray:
        if "expected_scores" in keys:
            return np.asarray(data["expected_scores"])
        if "expected_malware_probability" in keys:
            return np.asarray(data["expected_malware_probability"])
        raise KeyError(f"{npz_path} missing expected_scores / expected_malware_probability")

    if "headers" in keys and "bows" in keys:
        out: dict = {
            "headers": np.asarray(data["headers"], dtype=np.float32).tolist(),
            "bows": np.asarray(data["bows"], dtype=np.float32).tolist(),
        }
        scores = _scores_array().tolist()
        out["expected_scores"] = scores
    elif "vectors" in keys:
        out = {"vectors": np.asarray(data["vectors"], dtype=np.float32).tolist()}
        scores = _scores_array().tolist()
        out["expected_scores"] = scores
        out["expected_malware_probability"] = scores
    else:
        raise KeyError(f"{npz_path} has unsupported keys: {sorted(keys)}")

    if "labels" in keys:
        out["labels"] = np.asarray(data["labels"]).tolist()
    if "sample_ids" in keys:
        out["sample_ids"] = [str(x) for x in data["sample_ids"].tolist()]
    elif "indices" in keys:
        out["sample_ids"] = [str(int(x)) for x in data["indices"].tolist()]
    else:
        n = len(out.get("vectors") or out.get("headers") or [])
        out["sample_ids"] = [f"sample_{i:03d}" for i in range(n)]

    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "parity_dir",
        type=Path,
        help="Directory containing sample_vectors.npz (e.g. .../parity_samples)",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Output JSON path (default: <parity_dir>/parity_vectors.json)",
    )
    args = parser.parse_args(argv)

    parity_dir = args.parity_dir.resolve()
    out_path = (args.out or parity_dir / "parity_vectors.json").resolve()
    payload = build_parity_vectors(parity_dir)
    out_path.write_text(json.dumps(payload) + "\n", encoding="utf-8")
    print(out_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
