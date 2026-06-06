#!/usr/bin/env python3
"""Export MLDP ONNX bundle."""

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
from src.models.tiny_mlp import LinearSigmoidModule, TinyMlpModule

ONNX_OPSET = 14


def _build_torch_model(ckpt: dict) -> torch.nn.Module:
    feature_dim = int(ckpt["feature_dim"])
    if ckpt.get("model_type") == "tiny_mlp":
        model = TinyMlpModule(feature_dim, hidden_dim=int(ckpt.get("hidden_dim", 32)))
    else:
        model = LinearSigmoidModule(feature_dim)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model


def export_bundle(checkpoint: Path, out_dir: Path, cfg, num_parity: int) -> Path:
    ckpt = torch.load(checkpoint, map_location="cpu", weights_only=False)
    model = _build_torch_model(ckpt)
    out_dir.mkdir(parents=True, exist_ok=True)

    feature_dim = int(ckpt["feature_dim"])
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
        dynamic_axes={"permissions": {0: "batch_size"}, "malware_probability": {0: "batch_size"}},
    )

    features_dir = out_dir / "features"
    features_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(cfg.paths.selected_permissions, features_dir / "selected_permissions.json")
    rules_src = cfg.paths.mldp_dir / "association_rules.json"
    if rules_src.is_file():
        shutil.copy2(rules_src, out_dir / "mldp_rules.json")

    threshold = float(cfg.evaluation.get("threshold", 0.5))
    (out_dir / "thresholds.json").write_text(
        json.dumps(
            {
                "malware_threshold": threshold,
                "description": "Predict malware when malware_probability >= malware_threshold",
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    manifest = {
        "model_id": MODEL_ID,
        "domain": DOMAIN_ID,
        "exported_at": datetime.now(timezone.utc).isoformat(),
        "opset_version": ONNX_OPSET,
        "feature_dim": feature_dim,
        "model_type": ckpt.get("model_type"),
        "token_normalization": "vigidroid",
        "inputs": [{"name": "permissions", "shape": [1, feature_dim], "dtype": "float32"}],
        "outputs": [{"name": "malware_probability", "dtype": "float32"}],
        "android_assets_target": cfg.pipeline.get("android_assets_target"),
    }
    (out_dir / "export_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    X, y = stack_split_arrays(cfg.paths.processed, "val")
    n = min(num_parity, X.shape[0])
    rng = np.random.default_rng(42)
    idx = rng.choice(X.shape[0], size=n, replace=False)
    vectors, expected, labels = [], [], []
    with torch.no_grad():
        for i in idx:
            vec = torch.from_numpy(X[int(i)].astype(np.float32)).unsqueeze(0)
            score = float(model(vec).item())
            vectors.append(X[int(i)].astype(np.float32))
            expected.append(score)
            labels.append(int(y[int(i)]))

    samples_dir = out_dir / "parity_samples"
    samples_dir.mkdir(parents=True, exist_ok=True)
    np.savez(
        samples_dir / "sample_vectors.npz",
        vectors=np.stack(vectors),
        expected_malware_probability=np.asarray(expected),
        labels=np.asarray(labels, dtype=np.int64),
    )
    return out_dir


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--checkpoint", type=Path, default=ROOT / "artifacts/checkpoints/mldp_pruned.pth")
    parser.add_argument("--out-dir", type=Path, default=ROOT / "artifacts/export/mldp_pruned_permission")
    parser.add_argument("--num-parity-samples", type=int, default=10)
    args = parser.parse_args(argv)

    cfg = load_config(args.config)
    out = export_bundle(
        args.checkpoint.resolve(),
        args.out_dir.resolve(),
        cfg,
        args.num_parity_samples,
    )
    print(f"Export bundle → {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
