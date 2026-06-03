#!/usr/bin/env python3
"""Phase 3: export BM1 MLP(H) to ONNX deployment bundle (P7)."""

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
from src.data.dataloaders import resolve_processed_path
from src.data.store import load_processed_bundle
from src.features.dex_header import FEATURE_DIM
from src.features.multidex import multidex_settings
from src.models.mlp_header import build_mlp_header
from src.training.checkpoint import load_checkpoint
from src.training.run_logging import MODEL_ID, DOMAIN_ID, write_json

ONNX_OPSET = 14
DEFAULT_OUT = ROOT / "artifacts" / "export" / "mlp_header"


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def load_model_from_checkpoint(
    checkpoint_path: Path,
    *,
    config_path: Path | None = None,
) -> tuple[torch.nn.Module, dict[str, Any], Any]:
    cfg = load_config(config_path)
    ckpt = load_checkpoint(checkpoint_path, map_location="cpu")
    if ckpt is None:
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    feature_dim = int(ckpt.get("feature_dim", FEATURE_DIM))
    hidden_dim = int(ckpt.get("hidden_dim", cfg.model.get("hidden_dim", 128)))
    model = build_mlp_header(input_dim=feature_dim, hidden_dim=hidden_dim)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    meta = {
        "feature_dim": feature_dim,
        "hidden_dim": hidden_dim,
        "checkpoint": str(checkpoint_path.resolve()),
    }
    return model, meta, cfg


def export_onnx_model(
    model: torch.nn.Module,
    onnx_path: Path,
    *,
    feature_dim: int,
) -> None:
    onnx_path.parent.mkdir(parents=True, exist_ok=True)
    dummy = torch.zeros(1, feature_dim, dtype=torch.float32)
    # Legacy exporter (dynamo=False) yields stable opset 14 graphs for ORT mobile.
    torch.onnx.export(
        model,
        dummy,
        str(onnx_path),
        export_params=True,
        opset_version=ONNX_OPSET,
        do_constant_folding=True,
        dynamo=False,
        input_names=["features"],
        output_names=["malware_probability"],
        dynamic_axes={
            "features": {0: "batch_size"},
            "malware_probability": {0: "batch_size"},
        },
    )
    # Remove external .data sidecar if legacy export inlined weights.
    data_sidecar = onnx_path.with_suffix(".onnx.data")
    if data_sidecar.is_file():
        data_sidecar.unlink()


def build_export_manifest(
    cfg,
    meta: dict[str, Any],
    *,
    onnx_path: Path,
    normalization_rel: str,
) -> dict[str, Any]:
    md = multidex_settings(cfg.preprocessing)
    pre = cfg.preprocessing
    return {
        "model_id": MODEL_ID,
        "domain": DOMAIN_ID,
        "exported_at": _utc_now(),
        "opset_version": ONNX_OPSET,
        "checkpoint": meta["checkpoint"],
        "feature_dim": meta["feature_dim"],
        "hidden_dim": meta["hidden_dim"],
        "preprocessing_version": int(pre.get("cache_version", 2)),
        "multidex_mode": md["mode"],
        "multidex_max_dex": md["max_dex"],
        "normalization": normalization_rel,
        "inputs": [
            {
                "name": "features",
                "shape": [1, meta["feature_dim"]],
                "dtype": "float32",
                "description": "Min-max normalized Dex header vector (post-multidex aggregation)",
            }
        ],
        "outputs": [
            {
                "name": "malware_probability",
                "shape": [1, 1],
                "dtype": "float32",
                "description": "Sigmoid malware probability in [0, 1]",
            }
        ],
        "onnx_file": onnx_path.name,
        "android_assets_target": "vigidroid/app/src/main/assets/models/mlp_header/",
    }


def build_thresholds(cfg) -> dict[str, Any]:
    threshold = float(cfg.evaluation.get("threshold", 0.5))
    return {
        "malware_threshold": threshold,
        "benign_threshold": 1.0 - threshold,
        "description": "Predict malware when malware_probability >= malware_threshold",
    }


def write_parity_samples(
    model: torch.nn.Module,
    out_dir: Path,
    *,
    processed_path: Path,
    num_samples: int = 8,
    seed: int = 42,
) -> Path:
    bundle = load_processed_bundle(processed_path)
    n = bundle.features.shape[0]
    num_samples = min(num_samples, n)
    rng = np.random.default_rng(seed)
    indices = rng.choice(n, size=num_samples, replace=False)

    vectors: list[np.ndarray] = []
    labels: list[int] = []
    expected: list[float] = []
    sample_ids: list[str] = []

    with torch.no_grad():
        for i, idx in enumerate(indices):
            vec = bundle.features[int(idx)].float().unsqueeze(0)
            score = float(model(vec).item())
            vectors.append(vec.numpy().astype(np.float32).ravel())
            labels.append(int(bundle.labels[int(idx)].item()))
            expected.append(score)
            sample_ids.append(f"sample_{i:03d}")

    samples_dir = out_dir / "parity_samples"
    samples_dir.mkdir(parents=True, exist_ok=True)

    np.savez(
        samples_dir / "sample_vectors.npz",
        indices=indices.astype(np.int64),
        vectors=np.stack(vectors, axis=0),
        labels=np.asarray(labels, dtype=np.int64),
        expected_scores=np.asarray(expected, dtype=np.float64),
        sample_ids=np.asarray(sample_ids),
    )

    index = [
        {
            "sample_id": sid,
            "index": int(indices[i]),
            "label": labels[i],
            "expected_malware_probability": expected[i],
            "vector_dim": len(vectors[i]),
        }
        for i, sid in enumerate(sample_ids)
    ]
    write_json(samples_dir / "index.json", {"num_samples": num_samples, "seed": seed, "samples": index})
    return samples_dir


def copy_to_archive(export_dir: Path, archive_export: Path) -> None:
    archive_export.mkdir(parents=True, exist_ok=True)
    for path in export_dir.rglob("*"):
        if path.is_file():
            rel = path.relative_to(export_dir)
            dest = archive_export / rel
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, dest)


def verify_onnx(onnx_path: Path, feature_dim: int) -> dict[str, Any]:
    try:
        import onnxruntime as ort
    except ImportError as exc:
        raise SystemExit("onnxruntime required: pip install onnxruntime") from exc

    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    inp_name = session.get_inputs()[0].name
    out_name = session.get_outputs()[0].name
    x = np.zeros((1, feature_dim), dtype=np.float32)
    out = session.run([out_name], {inp_name: x})[0]
    return {
        "input_name": inp_name,
        "output_name": out_name,
        "test_output_shape": list(out.shape),
        "test_output_value": float(out.ravel()[0]),
    }


def export_bundle(
    *,
    checkpoint: Path,
    out_dir: Path,
    config_path: Path | None = None,
    archive_dir: Path | None = None,
    num_parity_samples: int = 8,
    skip_verify: bool = False,
) -> Path:
    model, meta, cfg = load_model_from_checkpoint(checkpoint, config_path=config_path)
    out_dir = out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    onnx_path = out_dir / "model.onnx"
    export_onnx_model(model, onnx_path, feature_dim=meta["feature_dim"])

    norm_src = cfg.paths.normalization_stats
    norm_dest = out_dir / "features" / "normalization_header.json"
    norm_dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(norm_src, norm_dest)

    write_json(out_dir / "thresholds.json", build_thresholds(cfg))
    write_json(
        out_dir / "export_manifest.json",
        build_export_manifest(
            cfg,
            meta,
            onnx_path=onnx_path,
            normalization_rel="features/normalization_header.json",
        ),
    )

    processed_path = resolve_processed_path(cfg)
    write_parity_samples(
        model,
        out_dir,
        processed_path=processed_path,
        num_samples=num_parity_samples,
    )

    if not skip_verify:
        verify_info = verify_onnx(onnx_path, meta["feature_dim"])
        manifest_path = out_dir / "export_manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest["onnx_runtime_check"] = verify_info
        write_json(manifest_path, manifest)

    if archive_dir is not None:
        copy_to_archive(out_dir, archive_dir / "export")

    return out_dir


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Export BM1 MLP(H) ONNX bundle.")
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=ROOT / "artifacts" / "checkpoints" / "latest_checkpoint.pth",
    )
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument(
        "--archive-dir",
        type=Path,
        default=None,
        help="Copy bundle to output_archives/<run>/export/ (default: LATEST_RUN.txt)",
    )
    parser.add_argument("--num-parity-samples", type=int, default=8)
    parser.add_argument("--skip-verify", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)

    archive_dir = args.archive_dir
    if archive_dir is None:
        latest = ROOT / "output_archives" / "LATEST_RUN.txt"
        if latest.is_file():
            run_id = latest.read_text(encoding="utf-8").strip()
            archive_dir = ROOT / "output_archives" / run_id

    out_dir = export_bundle(
        checkpoint=args.checkpoint.resolve(),
        out_dir=args.out_dir,
        config_path=args.config,
        archive_dir=archive_dir,
        num_parity_samples=args.num_parity_samples,
        skip_verify=args.skip_verify,
    )
    print(f"Export bundle written to {out_dir}")
    if archive_dir is not None:
        print(f"Copied to {archive_dir / 'export'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
