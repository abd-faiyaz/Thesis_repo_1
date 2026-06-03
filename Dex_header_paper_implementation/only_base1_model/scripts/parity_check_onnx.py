#!/usr/bin/env python3
"""Phase 4: PyTorch vs ONNX Runtime parity on export parity_samples (P8)."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.config import load_config
from src.models.mlp_header import build_mlp_header
from src.training.checkpoint import load_checkpoint
from src.training.run_logging import MODEL_ID, write_json

DEFAULT_BUNDLE = ROOT / "artifacts" / "export" / "mlp_header"
DEFAULT_TOLERANCE = 1e-4


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def load_pytorch_model(checkpoint: Path, *, config_path: Path | None = None) -> torch.nn.Module:
    cfg = load_config(config_path)
    ckpt = load_checkpoint(checkpoint, map_location="cpu")
    if ckpt is None:
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")

    feature_dim = int(ckpt.get("feature_dim", 104))
    hidden_dim = int(ckpt.get("hidden_dim", cfg.model.get("hidden_dim", 128)))
    model = build_mlp_header(input_dim=feature_dim, hidden_dim=hidden_dim)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model


def run_onnx(session, vectors: np.ndarray) -> np.ndarray:
    import onnxruntime as ort

    inp_name = session.get_inputs()[0].name
    out_name = session.get_outputs()[0].name
    scores: list[float] = []
    for i in range(vectors.shape[0]):
        x = vectors[i : i + 1].astype(np.float32)
        out = session.run([out_name], {inp_name: x})[0]
        scores.append(float(np.asarray(out).ravel()[0]))
    return np.asarray(scores, dtype=np.float64)


@torch.no_grad()
def run_pytorch(model: torch.nn.Module, vectors: np.ndarray) -> np.ndarray:
    scores: list[float] = []
    for i in range(vectors.shape[0]):
        x = torch.from_numpy(vectors[i : i + 1]).float()
        scores.append(float(model(x).item()))
    return np.asarray(scores, dtype=np.float64)


def load_parity_vectors(bundle_dir: Path) -> tuple[np.ndarray, np.ndarray, list[str]]:
    npz_path = bundle_dir / "parity_samples" / "sample_vectors.npz"
    if not npz_path.is_file():
        raise FileNotFoundError(f"Missing {npz_path}")
    data = np.load(npz_path)
    vectors = np.asarray(data["vectors"], dtype=np.float64)
    expected = np.asarray(data["expected_scores"], dtype=np.float64)
    sample_ids = [str(s) for s in data["sample_ids"].tolist()]
    return vectors, expected, sample_ids


def run_parity_check(
    bundle_dir: Path,
    *,
    checkpoint: Path,
    tolerance: float = DEFAULT_TOLERANCE,
    config_path: Path | None = None,
) -> dict[str, Any]:
    import onnxruntime as ort

    bundle_dir = bundle_dir.resolve()
    manifest_path = bundle_dir / "export_manifest.json"
    onnx_path = bundle_dir / "model.onnx"
    if not manifest_path.is_file() or not onnx_path.is_file():
        raise FileNotFoundError(f"Incomplete export bundle: {bundle_dir}")

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    vectors, export_expected, sample_ids = load_parity_vectors(bundle_dir)

    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    onnx_scores = run_onnx(session, vectors)
    model = load_pytorch_model(checkpoint, config_path=config_path)
    pytorch_scores = run_pytorch(model, vectors)

    pt_onnx_diff = np.abs(pytorch_scores - onnx_scores)
    onnx_export_diff = np.abs(onnx_scores - export_expected)

    per_sample = []
    for i, sid in enumerate(sample_ids):
        per_sample.append(
            {
                "sample_id": sid,
                "pytorch": float(pytorch_scores[i]),
                "onnx": float(onnx_scores[i]),
                "export_expected": float(export_expected[i]),
                "abs_diff_pytorch_onnx": float(pt_onnx_diff[i]),
                "abs_diff_onnx_export": float(onnx_export_diff[i]),
            }
        )

    max_pt_onnx = float(pt_onnx_diff.max())
    passed = max_pt_onnx < tolerance

    report: dict[str, Any] = {
        "timestamp": _utc_now(),
        "model_id": manifest.get("model_id", MODEL_ID),
        "bundle_dir": str(bundle_dir),
        "checkpoint": str(checkpoint.resolve()),
        "onnx_file": str(onnx_path),
        "tolerance": tolerance,
        "passed": passed,
        "n_samples": int(vectors.shape[0]),
        "pytorch_vs_onnx": {
            "max_abs_diff": max_pt_onnx,
            "mean_abs_diff": float(pt_onnx_diff.mean()),
            "per_sample": per_sample,
        },
        "onnx_vs_export_expected": {
            "max_abs_diff": float(onnx_export_diff.max()),
            "mean_abs_diff": float(onnx_export_diff.mean()),
        },
    }
    return report


def write_parity_outputs(
    report: dict[str, Any],
    bundle_dir: Path,
    *,
    archive_dir: Path | None,
    local_parity_dir: Path | None = None,
) -> tuple[Path, Path]:
    local = local_parity_dir or (ROOT / "artifacts" / "parity")
    local.mkdir(parents=True, exist_ok=True)

    report_path = local / "parity_report.json"
    write_json(report_path, report)

    npz_src = bundle_dir / "parity_samples" / "sample_vectors.npz"
    npz_dest = local / "sample_vectors.npz"
    shutil.copy2(npz_src, npz_dest)

    if archive_dir is not None:
        archive_parity = archive_dir / "parity"
        archive_parity.mkdir(parents=True, exist_ok=True)
        shutil.copy2(report_path, archive_parity / "parity_report.json")
        shutil.copy2(npz_dest, archive_parity / "sample_vectors.npz")

    return report_path, npz_dest


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="BM1 PyTorch vs ONNX parity check.")
    parser.add_argument(
        "--bundle",
        type=Path,
        default=DEFAULT_BUNDLE,
        help="Export bundle with model.onnx and parity_samples/",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=ROOT / "artifacts" / "checkpoints" / "latest_checkpoint.pth",
    )
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--tolerance", type=float, default=DEFAULT_TOLERANCE)
    parser.add_argument("--archive-dir", type=Path, default=None)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)

    archive_dir = args.archive_dir
    if archive_dir is None:
        latest = ROOT / "output_archives" / "LATEST_RUN.txt"
        if latest.is_file():
            archive_dir = ROOT / "output_archives" / latest.read_text(encoding="utf-8").strip()

    report = run_parity_check(
        args.bundle,
        checkpoint=args.checkpoint.resolve(),
        tolerance=args.tolerance,
        config_path=args.config,
    )
    report_path, npz_path = write_parity_outputs(
        report, args.bundle, archive_dir=archive_dir
    )

    status = "PASS" if report["passed"] else "FAIL"
    pt_onnx = report["pytorch_vs_onnx"]
    print(f"Parity {status}: max_abs_diff={pt_onnx['max_abs_diff']:.2e} (tolerance {args.tolerance})")
    print(f"  report → {report_path}")
    print(f"  vectors → {npz_path}")
    if archive_dir is not None:
        print(f"  archive → {archive_dir / 'parity'}")

    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
