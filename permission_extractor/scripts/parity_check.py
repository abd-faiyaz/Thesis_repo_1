#!/usr/bin/env python3
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.config import load_config
from src.models.tiny_mlp import LinearSigmoidModule, TinyMlpModule

TOLERANCE = 1e-4


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--export-dir", type=Path, default=ROOT / "artifacts/export/mldp_pruned_permission")
    parser.add_argument("--checkpoint", type=Path, default=ROOT / "artifacts/checkpoints/mldp_pruned.pth")
    args = parser.parse_args(argv)

    cfg = load_config(args.config)
    ckpt = torch.load(args.checkpoint.resolve(), map_location="cpu", weights_only=False)
    feature_dim = int(ckpt["feature_dim"])
    if ckpt.get("model_type") == "tiny_mlp":
        model = TinyMlpModule(feature_dim, hidden_dim=int(ckpt.get("hidden_dim", 32)))
    else:
        model = LinearSigmoidModule(feature_dim)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    samples = args.export_dir / "parity_samples" / "sample_vectors.npz"
    onnx_path = args.export_dir / "model.onnx"
    if not samples.is_file() or not onnx_path.is_file():
        print("Run export_onnx.py first.")
        return 1

    data = np.load(samples)
    vectors = data["vectors"].astype(np.float32)
    expected = data["expected_malware_probability"]

    with torch.no_grad():
        torch_scores = model(torch.from_numpy(vectors)).numpy().reshape(-1)

    import onnxruntime as ort

    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    inp = session.get_inputs()[0].name
    out = session.get_outputs()[0].name
    ort_scores = session.run([out], {inp: vectors})[0].reshape(-1)

    max_onnx = float(np.max(np.abs(torch_scores - ort_scores)))
    max_exp = float(np.max(np.abs(torch_scores - expected)))
    passed = max_onnx <= TOLERANCE and max_exp <= TOLERANCE

    report = {
        "max_delta_torch_vs_onnx": max_onnx,
        "max_delta_torch_vs_expected": max_exp,
        "passed": passed,
    }
    out_path = cfg.paths.artifacts / "metrics" / "parity_report.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
