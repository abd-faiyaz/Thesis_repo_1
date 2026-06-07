#!/usr/bin/env python3
"""Compare sklearn/PyTorch vs ONNX Runtime on parity samples."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.config import load_config
from src.models.mlr import LinRegDroidModule

TOLERANCE = 1e-4


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="LinRegDroid ONNX parity check.")
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--export-dir", type=Path, default=ROOT / "artifacts/export/linregdroid_permission")
    parser.add_argument("--checkpoint", type=Path, default=ROOT / "artifacts/checkpoints/linregdroid.pth")
    args = parser.parse_args(argv)

    cfg = load_config(args.config)
    export_dir = args.export_dir.resolve()
    onnx_path = export_dir / "model.onnx"
    samples_path = export_dir / "parity_samples" / "sample_vectors.npz"
    if not onnx_path.is_file() or not samples_path.is_file():
        print("Missing export bundle or parity samples; run export_onnx.py first.")
        return 1

    ckpt = torch.load(args.checkpoint.resolve(), map_location="cpu", weights_only=False)
    model = LinRegDroidModule(int(ckpt["feature_dim"]))
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    data = np.load(samples_path)
    vectors = data["vectors"].astype(np.float32)
    expected = data["expected_malware_probability"].astype(np.float64)

    with torch.no_grad():
        torch_scores = model(torch.from_numpy(vectors)).numpy().reshape(-1)

    try:
        import onnxruntime as ort
    except ImportError:
        print("Install onnxruntime for parity check.")
        return 1

    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    inp_name = session.get_inputs()[0].name
    out_name = session.get_outputs()[0].name
    ort_scores = session.run([out_name], {inp_name: vectors})[0].reshape(-1)

    max_torch_onnx = float(np.max(np.abs(torch_scores - ort_scores)))
    max_torch_expected = float(np.max(np.abs(torch_scores - expected)))
    ok = max_torch_onnx <= TOLERANCE and max_torch_expected <= TOLERANCE

    report = {
        "tolerance": TOLERANCE,
        "max_delta_torch_vs_onnx": max_torch_onnx,
        "max_delta_torch_vs_expected": max_torch_expected,
        "num_samples": int(len(vectors)),
        "passed": ok,
    }
    out_path = cfg.paths.artifacts / "metrics" / "parity_report.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    try:
        from src.thesis_archive import after_parity

        after_parity(out_path)
    except ImportError:
        pass

    print(json.dumps(report, indent=2))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
