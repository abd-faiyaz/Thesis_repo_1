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

from shared_calibration import build_val_thresholds_payload, write_export_thresholds

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

    default_threshold = float(cfg.evaluation.get("threshold", 0.5))
    primary_variant = str(cfg.model.get("variant", "linregdroid1"))
    write_export_thresholds(
        cfg.paths.artifacts / "metrics" / "thresholds.json",
        out_dir / "thresholds.json",
        fallback=build_val_thresholds_payload(
            model_id=MODEL_ID,
            y_true=np.array([0, 1]),
            scores=np.array([0.25, 0.75]),
            default=default_threshold,
            tune=False,
            calibrate_bands=False,
            cascade_targets=cfg.raw.get("cascade", {}),
            extra={
                "decision_variant": primary_variant,
                "description": (
                    "LinRegDroid1: predict malware when malware_probability >= tuned_val. "
                    "LinRegDroid2: nearest-class on raw linear score (ignores tuned_val)."
                ),
            },
        ),
    )
    threshold = float(
        json.loads((out_dir / "thresholds.json").read_text(encoding="utf-8"))["tuned_val"]
    )

    manifest = {
        "model_id": MODEL_ID,
        "domain": DOMAIN_ID,
        "exported_at": _utc_now(),
        "opset_version": ONNX_OPSET,
        "preprocessing_version": "1",
        "multidex_mode": "n/a",
        "split_mode": cfg.preprocessing.get("split_mode", "temporal_holdout"),
        "train_years": cfg.preprocessing.get("train_years", [2020, 2021]),
        "holdout_years": cfg.preprocessing.get("holdout_years", [2022, 2023]),
        "primary_test_split": "test",
        "feature_dim": feature_dim,
        "token_normalization": "vigidroid",
        "label_encoding": {
            "thesis": "benign=0, malware=1",
            "paper": "benign=1, malware=0",
            "note": "Training uses thesis labels; LinRegDroid2 applies paper nearest-class geometry at inference.",
        },
        "decision_variant": primary_variant,
        "decision_rules": {
            "linregdroid1": f"malware if clamp(linear_score, 0, 1) >= {threshold}",
            "linregdroid2": "malware if |ŷ - 0| > |ŷ - 1| in paper label space (mapped to thesis labels)",
        },
        "score_output": {
            "onnx_name": "malware_probability",
            "transform": "clamp(linear_score, 0, 1)",
            "paper_note": "Paper reports unbounded ŷ; mobile ONNX uses clamped probability for stable thresholds.",
            "raw_score_export": False,
        },
        "ensemble_scope": cfg.model.get("ensemble_scope", "out_of_scope"),
        "ensemble_note": (
            "Paper Ensemble-1 (5× MLR bootstrap) and Ensemble-2 (MLR+SVM+trees) not implemented; "
            "single MLR baseline for thesis comparison."
        ),
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
        "feature_asset": "features/permission_vocab.json",
        "android_assets_target": cfg.pipeline.get(
            "android_assets_target",
            "vigidroid/app/src/main/assets/models/linregdroid_permission/",
        ),
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
    try:
        import sys
        sys.path.insert(0, str(ROOT))
        from src.thesis_archive import after_export

        after_export()
    except ImportError:
        pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
