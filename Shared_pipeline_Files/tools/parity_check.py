#!/usr/bin/env python3
"""ONNX vs parity_samples expected scores (repo-level P8 helper)."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def load_parity_samples(bundle_dir: Path) -> tuple[np.ndarray, np.ndarray, list[str]]:
    npz_path = bundle_dir / "parity_samples" / "sample_vectors.npz"
    if not npz_path.is_file():
        raise FileNotFoundError(f"Missing {npz_path}")

    data = np.load(npz_path)
    vectors = np.asarray(data["vectors"], dtype=np.float32)

    if "expected_malware_probability" in data:
        expected = np.asarray(data["expected_malware_probability"], dtype=np.float64)
    elif "expected_scores" in data:
        expected = np.asarray(data["expected_scores"], dtype=np.float64)
    else:
        raise KeyError(f"No expected scores in {npz_path}")

    if "sample_ids" in data:
        sample_ids = [str(s) for s in data["sample_ids"].tolist()]
    elif "indices" in data:
        sample_ids = [str(int(i)) for i in data["indices"].tolist()]
    else:
        sample_ids = [str(i) for i in range(vectors.shape[0])]

    return vectors, expected, sample_ids


def run_onnx(bundle_dir: Path, vectors: np.ndarray) -> np.ndarray:
    import onnxruntime as ort

    onnx_path = bundle_dir / "model.onnx"
    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    inp_name = session.get_inputs()[0].name
    out_name = session.get_outputs()[0].name

    scores: list[float] = []
    for i in range(vectors.shape[0]):
        x = vectors[i : i + 1].astype(np.float32)
        out = session.run([out_name], {inp_name: x})[0]
        scores.append(float(np.asarray(out).ravel()[0]))
    return np.asarray(scores, dtype=np.float64)


def run_parity_check(bundle_dir: Path, *, tolerance: float = 1e-4) -> dict[str, Any]:
    bundle_dir = bundle_dir.resolve()
    manifest_path = bundle_dir / "export_manifest.json"
    onnx_path = bundle_dir / "model.onnx"
    if not manifest_path.is_file() or not onnx_path.is_file():
        raise FileNotFoundError(f"Incomplete export bundle: {bundle_dir}")

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    vectors, expected, sample_ids = load_parity_samples(bundle_dir)
    onnx_scores = run_onnx(bundle_dir, vectors)
    diffs = np.abs(onnx_scores - expected)

    per_sample = [
        {
            "sample_id": sample_ids[i],
            "onnx": float(onnx_scores[i]),
            "expected": float(expected[i]),
            "abs_diff": float(diffs[i]),
        }
        for i in range(len(sample_ids))
    ]

    max_diff = float(diffs.max()) if len(diffs) else 0.0
    passed = max_diff <= tolerance

    return {
        "timestamp": _utc_now(),
        "model_id": manifest.get("model_id"),
        "bundle_dir": str(bundle_dir),
        "onnx_file": str(onnx_path),
        "tolerance": tolerance,
        "passed": passed,
        "n_samples": int(vectors.shape[0]),
        "max_delta_onnx_vs_expected": max_diff,
        "mean_delta_onnx_vs_expected": float(diffs.mean()) if len(diffs) else 0.0,
        "per_sample": per_sample,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Compare ONNX output vs parity_samples expected scores.")
    parser.add_argument("--bundle", type=Path, required=True, help="artifacts/export/<model_id>/ directory")
    parser.add_argument("--tolerance", type=float, default=1e-4)
    parser.add_argument(
        "--report-out",
        type=Path,
        default=None,
        help="Optional path to write parity_report.json (default: print only)",
    )
    args = parser.parse_args()

    try:
        report = run_parity_check(args.bundle.resolve(), tolerance=args.tolerance)
    except FileNotFoundError as exc:
        print(exc, file=sys.stderr)
        return 1

    text = json.dumps(report, indent=2) + "\n"
    if args.report_out is not None:
        args.report_out.parent.mkdir(parents=True, exist_ok=True)
        args.report_out.write_text(text, encoding="utf-8")
        print(f"Wrote → {args.report_out}")

    status = "PASS" if report["passed"] else "FAIL"
    print(
        f"Parity {status}: model_id={report['model_id']} "
        f"max_delta={report['max_delta_onnx_vs_expected']:.2e} (tolerance {args.tolerance})"
    )
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
