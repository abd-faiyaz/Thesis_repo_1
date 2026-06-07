"""P8 — PyTorch vs ONNX Runtime parity on export parity_samples."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch

from src.config import PipelineConfig, ensure_artifact_dirs, load_config
from src.models.export_wrapper import MalwareProbExport
from src.models.factory import build_deployment_model_from_config
from src.training.checkpoint import load_best_checkpoint, restore_model_weights

_PACKAGE_ROOT = Path(__file__).resolve().parent.parent.parent


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def load_parity_vectors(bundle_dir: Path) -> tuple[np.ndarray, np.ndarray, list[str]]:
    npz_path = bundle_dir / "parity_samples" / "sample_vectors.npz"
    if not npz_path.is_file():
        raise FileNotFoundError(f"Missing parity vectors: {npz_path}")

    data = np.load(npz_path)
    vectors = np.asarray(data["vectors"], dtype=np.float32)
    if "expected_malware_probability" in data:
        expected = np.asarray(data["expected_malware_probability"], dtype=np.float64)
    else:
        expected = np.asarray(data["expected_scores"], dtype=np.float64)
    sample_ids = [str(s) for s in data["sample_ids"].tolist()]
    return vectors, expected, sample_ids


def load_pytorch_export_model(checkpoint: Path, cfg: PipelineConfig) -> MalwareProbExport:
    payload = load_best_checkpoint(checkpoint)
    input_dim = int(payload["d"])
    base = build_deployment_model_from_config(cfg, input_dim)
    restore_model_weights(base, payload)
    base.eval()
    export_model = MalwareProbExport(base)
    export_model.eval()
    return export_model


@torch.no_grad()
def run_pytorch(model: MalwareProbExport, vectors: np.ndarray) -> np.ndarray:
    scores: list[float] = []
    for i in range(vectors.shape[0]):
        x = torch.from_numpy(vectors[i : i + 1].astype(np.float32))
        scores.append(float(model(x).view(-1)[0].item()))
    return np.asarray(scores, dtype=np.float64)


def run_onnx(session, vectors: np.ndarray) -> np.ndarray:
    inp_name = session.get_inputs()[0].name
    out_name = session.get_outputs()[0].name
    scores: list[float] = []
    for i in range(vectors.shape[0]):
        x = vectors[i : i + 1].astype(np.float32)
        out = session.run([out_name], {inp_name: x})[0]
        scores.append(float(np.asarray(out, dtype=np.float64).ravel()[0]))
    return np.asarray(scores, dtype=np.float64)


def run_parity_check(
    cfg: PipelineConfig,
    *,
    bundle_dir: Path | None = None,
    checkpoint: Path | None = None,
    tolerance: float | None = None,
) -> dict[str, Any]:
    ensure_artifact_dirs(cfg)
    bundle = (bundle_dir or cfg.paths.export).resolve()
    ckpt = (checkpoint or (cfg.paths.checkpoints / "best.pt")).resolve()
    tol = float(tolerance if tolerance is not None else cfg.export.get("parity_max_delta", 1e-4))

    manifest_path = bundle / "export_manifest.json"
    onnx_path = bundle / "model.onnx"
    if not manifest_path.is_file() or not onnx_path.is_file():
        raise FileNotFoundError(f"Incomplete export bundle: {bundle}")
    if not ckpt.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt}")

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    vectors, export_expected, sample_ids = load_parity_vectors(bundle)

    import onnxruntime as ort

    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    model = load_pytorch_export_model(ckpt, cfg)

    pytorch_scores = run_pytorch(model, vectors)
    onnx_scores = run_onnx(session, vectors)

    pt_onnx_diff = np.abs(pytorch_scores - onnx_scores)
    pt_export_diff = np.abs(pytorch_scores - export_expected)
    onnx_export_diff = np.abs(onnx_scores - export_expected)

    per_sample: list[dict[str, Any]] = []
    for i, sid in enumerate(sample_ids):
        per_sample.append(
            {
                "sample_id": sid,
                "pytorch_malware_prob": float(pytorch_scores[i]),
                "onnx_malware_prob": float(onnx_scores[i]),
                "expected_malware_prob": float(export_expected[i]),
                "delta_pytorch_onnx": float(pt_onnx_diff[i]),
                "delta_pytorch_expected": float(pt_export_diff[i]),
                "delta_onnx_expected": float(onnx_export_diff[i]),
            }
        )

    max_delta = float(pt_onnx_diff.max())
    passed = max_delta <= tol

    return {
        "timestamp": _utc_now(),
        "model_id": manifest.get("model_id", cfg.model_id),
        "domain": manifest.get("domain", cfg.domain),
        "bundle_dir": str(bundle),
        "checkpoint": str(ckpt),
        "onnx_file": str(onnx_path),
        "tolerance": tol,
        "passed": passed,
        "max_delta": max_delta,
        "mean_delta": float(pt_onnx_diff.mean()),
        "n_samples": int(vectors.shape[0]),
        "pytorch_vs_onnx": {
            "max_delta": max_delta,
            "mean_delta": float(pt_onnx_diff.mean()),
            "per_sample": per_sample,
        },
        "pytorch_vs_export_expected": {
            "max_delta": float(pt_export_diff.max()),
            "mean_delta": float(pt_export_diff.mean()),
        },
        "onnx_vs_export_expected": {
            "max_delta": float(onnx_export_diff.max()),
            "mean_delta": float(onnx_export_diff.mean()),
        },
    }


def write_parity_report(cfg: PipelineConfig, report: dict[str, Any]) -> Path:
    out_path = cfg.paths.metrics / "parity_report.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return out_path


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="PyTorch vs ONNX parity check (P8).")
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--bundle", type=Path, default=None, help="Export bundle directory")
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help="Default: artifacts/checkpoints/best.pt",
    )
    parser.add_argument("--tolerance", type=float, default=None)
    return parser


def main(argv: list[str] | None = None) -> int:
    if str(_PACKAGE_ROOT) not in sys.path:
        sys.path.insert(0, str(_PACKAGE_ROOT))

    args = build_arg_parser().parse_args(argv)
    cfg = load_config(args.config)

    report = run_parity_check(
        cfg,
        bundle_dir=args.bundle,
        checkpoint=args.checkpoint,
        tolerance=args.tolerance,
    )
    report_path = write_parity_report(cfg, report)

    status = "PASS" if report["passed"] else "FAIL"
    print(f"P8 parity {status}: max_delta={report['max_delta']:.2e} (tolerance {report['tolerance']})")
    print(f"  samples: {report['n_samples']}")
    print(f"  report → {report_path}")
    try:
        from src.thesis_archive import after_parity

        after_parity(report_path)
    except ImportError:
        pass
    if not report["passed"]:
        worst = max(
            report["pytorch_vs_onnx"]["per_sample"],
            key=lambda row: row["delta_pytorch_onnx"],
        )
        print(
            f"  worst sample: {worst['sample_id']} "
            f"pytorch={worst['pytorch_malware_prob']:.6f} "
            f"onnx={worst['onnx_malware_prob']:.6f} "
            f"delta={worst['delta_pytorch_onnx']:.2e}"
        )
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
