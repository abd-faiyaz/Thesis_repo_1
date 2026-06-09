"""P8 — PyTorch vs ONNX parity on parity_samples."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch

from src.config import load_config
from src.models.export_wrapper import FusionMalwareProbExport
from src.training.checkpoint import load_best_checkpoint, restore_model_weights
from src.training.setup import build_fusion_model

_PACKAGE_ROOT = Path(__file__).resolve().parent.parent.parent


def parity_check(cfg, export_dir: Path, *, max_delta: float) -> dict:
    onnx_path = export_dir / "model.onnx"
    if not onnx_path.is_file():
        raise FileNotFoundError(onnx_path)

    payload = load_best_checkpoint(cfg.paths.checkpoints / "best.pt")
    receiver_dim = int(payload["receiver_dim"])
    dex_dim = int(payload.get("dex_dim", 104))
    model = build_fusion_model(cfg, dex_dim=dex_dim, receiver_dim=receiver_dim)
    restore_model_weights(model, payload)
    model.eval()
    export_model = FusionMalwareProbExport(model)
    export_model.eval()

    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    input_names = [inp.name for inp in session.get_inputs()]

    samples_dir = export_dir / "parity_samples"
    report: dict = {"samples": [], "max_delta": 0.0, "passed": True}

    for sample_dir in sorted(samples_dir.glob("sample_*")):
        H = np.load(sample_dir / "H.npy").astype(np.float32).reshape(1, -1)
        R = np.load(sample_dir / "R.npy").astype(np.float32).reshape(1, -1)
        expected = float(
            json.loads((sample_dir / "expected_prob.json").read_text())["malware_prob"]
        )

        with torch.no_grad():
            pt_prob = float(
                export_model(torch.from_numpy(H), torch.from_numpy(R)).item()
            )

        feeds: dict[str, np.ndarray] = {}
        for name in input_names:
            if name == "dex_header":
                feeds[name] = H
            elif name == "receiver":
                feeds[name] = R
        onnx_prob = float(session.run(None, feeds)[0].reshape(-1)[0])
        delta = abs(pt_prob - onnx_prob)
        exp_delta = abs(pt_prob - expected)
        report["max_delta"] = max(report["max_delta"], delta, exp_delta)
        ok = delta <= max_delta and exp_delta <= max_delta
        if not ok:
            report["passed"] = False
        report["samples"].append(
            {
                "sample": sample_dir.name,
                "pytorch": pt_prob,
                "onnx": onnx_prob,
                "expected": expected,
                "delta_onnx": delta,
                "delta_expected": exp_delta,
                "ok": ok,
            }
        )

    out = cfg.paths.metrics / "parity_report.json"
    out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="P8 ONNX parity.")
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--export-dir", type=Path, default=None)
    args = parser.parse_args(argv)

    if str(_PACKAGE_ROOT) not in sys.path:
        sys.path.insert(0, str(_PACKAGE_ROOT))

    cfg = load_config(args.config)
    export_dir = args.export_dir or cfg.paths.export
    max_delta = float(cfg.export.get("parity_max_delta", 1e-4))
    report = parity_check(cfg, export_dir, max_delta=max_delta)
    print(f"max_delta={report['max_delta']:.2e}  passed={report['passed']}")
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
