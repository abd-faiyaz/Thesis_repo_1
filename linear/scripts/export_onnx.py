#!/usr/bin/env python3
"""Export LinRegDroid ONNX bundle for future VigiDroid integration."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.config import load_config
from src.constants import DOMAIN_ID, MODEL_ID
from src.data.dataset import stack_split_arrays
from src.models.mlr import LinRegDroidModule

ONNX_OPSET = 14


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def export_bundle(
    *,
    checkpoint: Path,
    out_dir: Path,
    config_path: Path | None = None,
    num_parity_samples: int = 10,
) -> Path:
    cfg = load_config(config_path)
    ckpt = torch.load(checkpoint, map_location="cpu", weights_only=False)
    feature_dim = int(ckpt["feature_dim"])
    model = LinRegDroidModule(feature_dim)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    out_dir.mkdir(parents=True, exist_ok=True)
    onnx_path = out_dir / "model.onnx"
    dummy = torch.zeros(1, feature_dim, dtype=torch.float32)
    torch.onnx.export(
        model,
        dummy,
        str(onnx_path),
        export_params=True,
        opset_version=ONNX_OPSET,
        do_constant_folding=True,
        dynamo=False,
        input_names=["permissions"],
        output_names=["malware_probability"],
        dynamic_axes={
            "permissions": {0: "batch_size"},
            "malware_probability": {0: "batch_size"},
        },
    )
    data_sidecar = onnx_path.with_suffix(".onnx.data")
    if data_sidecar.is_file():
        data_sidecar.unlink()

    features_dir = out_dir / "features"
    features_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(cfg.paths.permission_vocab, features_dir / "permission_vocab.json")
    if (cfg.paths.checkpoints / "coefficients.json").is_file():
        shutil.copy2(cfg.paths.checkpoints / "coefficients.json", out_dir / "coefficients.json")

    threshold = float(cfg.evaluation.get("threshold", 0.5))
    thresholds = {
        "malware_threshold": threshold,
        "description": "Predict malware when malware_probability >= malware_threshold",
    }
    (out_dir / "thresholds.json").write_text(json.dumps(thresholds, indent=2) + "\n", encoding="utf-8")

    manifest = {
        "model_id": MODEL_ID,
        "domain": DOMAIN_ID,
        "exported_at": _utc_now(),
        "opset_version": ONNX_OPSET,
        "feature_dim": feature_dim,
        "token_normalization": "vigidroid",
        "inputs": [
            {
                "name": "permissions",
                "shape": [1, feature_dim],
                "dtype": "float32",
                "description": "Binary permission vector aligned with features/permission_vocab.json",
            }
        ],
        "outputs": [
            {
                "name": "malware_probability",
                "shape": [1, 1],
                "dtype": "float32",
                "description": "Malware probability in [0, 1] (clamp of linear score)",
            }
        ],
        "onnx_file": onnx_path.name,
        "android_assets_target": cfg.pipeline.get(
            "android_assets_target",
            "vigidroid/app/src/main/assets/models/linregdroid_permission/",
        ),
        "ensemble_note": "Designed to run alongside existing VigiDroid models; integration deferred.",
    }
    (out_dir / "export_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    _write_parity_samples(model, out_dir, cfg, num_parity_samples)
    return out_dir


def _write_parity_samples(model, out_dir: Path, cfg, num_samples: int) -> None:
    X, y = stack_split_arrays(cfg.paths.processed, "val")
    n = min(num_samples, X.shape[0])
    rng = np.random.default_rng(42)
    indices = rng.choice(X.shape[0], size=n, replace=False)

    vectors = []
    expected = []
    labels = []
    with torch.no_grad():
        for idx in indices:
            vec = torch.from_numpy(X[int(idx)]).float().unsqueeze(0)
            score = float(model(vec).item())
            vectors.append(X[int(idx)].astype(np.float32))
            expected.append(score)
            labels.append(int(y[int(idx)]))

    samples_dir = out_dir / "parity_samples"
    samples_dir.mkdir(parents=True, exist_ok=True)
    np.savez(
        samples_dir / "sample_vectors.npz",
        indices=indices.astype(np.int64),
        vectors=np.stack(vectors, axis=0),
        labels=np.asarray(labels, dtype=np.int64),
        expected_malware_probability=np.asarray(expected, dtype=np.float64),
    )
    index = [
        {
            "index": int(indices[i]),
            "label": labels[i],
            "expected_malware_probability": expected[i],
        }
        for i in range(n)
    ]
    (samples_dir / "index.json").write_text(
        json.dumps({"num_samples": n, "samples": index}, indent=2) + "\n",
        encoding="utf-8",
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Export LinRegDroid ONNX bundle.")
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--checkpoint", type=Path, default=ROOT / "artifacts/checkpoints/linregdroid.pth")
    parser.add_argument("--out-dir", type=Path, default=ROOT / "artifacts/export/linregdroid_permission")
    parser.add_argument("--num-parity-samples", type=int, default=10)
    args = parser.parse_args(argv)

    out = export_bundle(
        checkpoint=args.checkpoint.resolve(),
        out_dir=args.out_dir.resolve(),
        config_path=args.config,
        num_parity_samples=args.num_parity_samples,
    )
    print(f"Export bundle → {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
