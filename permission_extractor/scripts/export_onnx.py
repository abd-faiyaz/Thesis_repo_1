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
    # mldp_rules.json stays under artifacts/mldp/ (analysis only) — not shipped to mobile.

    selected_meta = {}
    if cfg.paths.selected_permissions.is_file():
        selected_meta = json.loads(cfg.paths.selected_permissions.read_text(encoding="utf-8"))

    threshold = float(cfg.evaluation.get("threshold", 0.5))
    model_type = ckpt.get("model_type", "linear_svc")
    (out_dir / "thresholds.json").write_text(
        json.dumps(
            {
                "malware_threshold": threshold,
                "model_type": model_type,
                "description": (
                    f"Predict malware when malware_probability >= malware_threshold "
                    f"(exported {model_type})"
                ),
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
        "preprocessing_version": "1",
        "multidex_mode": "n/a",
        "split_mode": cfg.preprocessing.get("split_mode", "stratified_development"),
        "train_years": cfg.preprocessing.get("development_years", [2020, 2021]),
        "test_years": cfg.preprocessing.get("temporal_holdout_years", [2022, 2023]),
        "primary_test_split": "temporal_holdout",
        "feature_dim": feature_dim,
        "model_type": model_type,
        "token_normalization": "vigidroid",
        "feature_asset": "features/selected_permissions.json",
        "mldp": {
            "frozen_set_s": feature_dim,
            "association_rule_mode": selected_meta.get(
                "association_rule_mode", cfg.mldp.get("association", {}).get("rule_mode")
            ),
            "association_rule_note": selected_meta.get(
                "association_rule_note",
                "Malware-only FP-Growth itemsets; implicit malware consequent (thesis simplification).",
            ),
            "fallback_used": selected_meta.get("fallback_used"),
            "full_permission_ablation": "not_implemented",
        },
        "inputs": [
            {
                "name": "permissions",
                "shape": [1, feature_dim],
                "dtype": "float32",
                "description": "Binary vector aligned with features/selected_permissions.json (frozen set S)",
            }
        ],
        "outputs": [
            {
                "name": "malware_probability",
                "shape": [1, 1],
                "dtype": "float32",
                "description": "Sigmoid-calibrated malware probability in [0, 1]",
            }
        ],
        "onnx_file": onnx_path.name,
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
        indices=idx.astype(np.int64),
        vectors=np.stack(vectors),
        expected_malware_probability=np.asarray(expected, dtype=np.float64),
        labels=np.asarray(labels, dtype=np.int64),
    )
    index = [
        {
            "index": int(idx[i]),
            "label": labels[i],
            "expected_malware_probability": expected[i],
        }
        for i in range(n)
    ]
    (samples_dir / "index.json").write_text(
        json.dumps({"num_samples": n, "samples": index}, indent=2) + "\n",
        encoding="utf-8",
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
    try:
        from src.thesis_archive import after_export

        after_export()
    except ImportError:
        pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
