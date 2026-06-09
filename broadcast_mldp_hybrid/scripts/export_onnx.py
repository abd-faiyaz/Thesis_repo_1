#!/usr/bin/env python3
"""P7 — export broadcast+MLDP hybrid ONNX deployment bundle."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.config import ensure_artifact_dirs, load_config
from src.data.store import feature_shard_path, load_feature_shard, load_preprocessing_meta
from src.models.export_wrapper import MalwareProbExport
from src.models.factory import build_deployment_model_from_config
from src.training.checkpoint import load_best_checkpoint, restore_model_weights

ANDROID_ASSETS = "vigidroid/app/src/main/assets/models/broadcast_mldp_hybrid/"


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def _git_revision(root: Path) -> str:
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=root,
            stderr=subprocess.DEVNULL,
            text=True,
        )
        return out.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def load_export_model(
    checkpoint_path: Path,
    cfg,
) -> tuple[MalwareProbExport, dict[str, Any]]:
    payload = load_best_checkpoint(checkpoint_path)
    input_dim = int(payload["d"])
    base = build_deployment_model_from_config(cfg, input_dim)
    restore_model_weights(base, payload)
    base.eval()
    export_model = MalwareProbExport(base)
    export_model.eval()
    meta = {
        "checkpoint": str(checkpoint_path.resolve()),
        "input_dim": input_dim,
        "S": len(payload["S"]),
        "R": len(payload["A"]),
        "deployment": payload.get("deployment", cfg.classifier.get("deployment")),
        "config_hash": payload.get("config_hash"),
        "val_metrics": payload.get("val_metrics", {}),
    }
    return export_model, meta


def export_onnx_model(
    model: MalwareProbExport,
    onnx_path: Path,
    *,
    feature_dim: int,
    opset: int,
) -> None:
    onnx_path.parent.mkdir(parents=True, exist_ok=True)
    dummy = torch.zeros(1, feature_dim, dtype=torch.float32)
    torch.onnx.export(
        model,
        dummy,
        str(onnx_path),
        export_params=True,
        opset_version=opset,
        do_constant_folding=True,
        dynamo=False,
        input_names=["features"],
        output_names=["malware_prob"],
        dynamic_axes={
            "features": {0: "batch_size"},
            "malware_prob": {0: "batch_size"},
        },
    )
    data_sidecar = onnx_path.with_suffix(".onnx.data")
    if data_sidecar.is_file():
        data_sidecar.unlink()


def copy_feature_assets(cfg, out_dir: Path) -> None:
    features_dir = out_dir / "features"
    features_dir.mkdir(parents=True, exist_ok=True)
    processed = cfg.paths.processed
    for name in (
        "mldp_permission_vocab.json",
        "receiver_action_vocab.json",
        "feature_layout.json",
    ):
        src = processed / name
        if not src.is_file():
            raise FileNotFoundError(f"Missing processed feature file: {src}")
        shutil.copy2(src, features_dir / name)

    system_src = cfg.paths.system_actions_file
    if not system_src.is_file():
        raise FileNotFoundError(f"Missing system_actions.json: {system_src}")
    shutil.copy2(system_src, features_dir / "system_actions.json")


def load_thresholds(cfg) -> dict[str, float]:
    from shared_calibration import build_val_thresholds_payload, read_thresholds_payload

    default = float(cfg.evaluation.get("threshold", 0.5))
    return read_thresholds_payload(
        cfg.paths.metrics / "thresholds.json",
        fallback=build_val_thresholds_payload(
            model_id=cfg.model_id,
            y_true=np.array([0, 1]),
            scores=np.array([0.25, 0.75]),
            default=default,
            tune=False,
            calibrate_bands=False,
            cascade_targets=cfg.raw.get("cascade", {}),
            extra={
                "model_type": str(cfg.classifier.get("deployment", "tiny_mlp")),
                "description": "Run evaluate to calibrate tuned_val and cascade bands",
            },
        ),
    )


def build_export_manifest(
    cfg,
    meta: dict[str, Any],
    *,
    opset: int,
    preprocessing_version: str,
) -> dict[str, Any]:
    layout_path = cfg.paths.processed / "feature_layout.json"
    layout = json.loads(layout_path.read_text(encoding="utf-8"))
    s_size = int(layout["S"])
    r_size = int(layout["R"])
    d = int(layout["total"])
    return {
        "model_id": cfg.model_id,
        "domain": cfg.domain,
        "exported_at": _utc_now(),
        "opset": opset,
        "checkpoint": meta["checkpoint"],
        "config_hash": meta.get("config_hash"),
        "preprocessing_version": preprocessing_version,
        "multidex_mode": "n/a",
        "feature_extraction": {
            "apk_part": "AndroidManifest.xml",
            "fusion": "early_concat",
            "blocks": ["mldp_perms", "receiver_system_actions"],
            "mldp_size_S": s_size,
            "receiver_size_R": r_size,
            "receiver_system_actions_only": bool(
                cfg.features.get("receiver_system_actions_only", True)
            ),
        },
        "inputs": [
            {
                "name": "features",
                "shape": [1, d],
                "dtype": "float32",
                "description": "Early-fused [x_S || x_R] binary vector, float32 0/1",
            }
        ],
        "outputs": [
            {
                "name": "malware_prob",
                "shape": [1, 1],
                "dtype": "float32",
                "description": "Malware probability in [0, 1] (sigmoid in ONNX graph)",
            }
        ],
        "onnx_file": "model.onnx",
        "android_assets_target": ANDROID_ASSETS,
    }


def write_parity_samples(
    model: MalwareProbExport,
    out_dir: Path,
    *,
    val_shard_path: Path,
    num_samples: int,
    seed: int,
) -> None:
    shard = load_feature_shard(val_shard_path, split="val")
    n = shard.x.shape[0]
    num_samples = min(num_samples, n)
    rng = np.random.default_rng(seed)
    indices = rng.choice(n, size=num_samples, replace=False)

    vectors: list[list[float]] = []
    expected: list[float] = []
    labels: list[int] = []
    sample_ids: list[str] = []
    index_rows: list[dict[str, Any]] = []

    with torch.no_grad():
        for i, idx in enumerate(indices):
            vec = shard.x[int(idx)].float().unsqueeze(0)
            prob = float(model(vec).view(-1)[0].item())
            vec_list = vec.numpy().astype(np.float32).ravel().tolist()
            label = int(shard.y[int(idx)].item())

            vectors.append(vec_list)
            expected.append(prob)
            labels.append(label)
            sid = f"sample_{i:03d}"
            sample_ids.append(sid)
            index_rows.append(
                {
                    "sample_id": sid,
                    "index": int(idx),
                    "label": label,
                    "expected_malware_probability": prob,
                    "vector_dim": len(vec_list),
                }
            )

            sample_dir = out_dir / "parity_samples" / sid
            sample_dir.mkdir(parents=True, exist_ok=True)
            np.save(sample_dir / "x.npy", np.asarray(vec_list, dtype=np.float32))
            _write_json(sample_dir / "expected_prob.json", {"malware_prob": prob})

    samples_dir = out_dir / "parity_samples"
    samples_dir.mkdir(parents=True, exist_ok=True)
    np.savez(
        samples_dir / "sample_vectors.npz",
        indices=indices.astype(np.int64),
        vectors=np.stack([np.asarray(v, dtype=np.float32) for v in vectors]),
        labels=np.asarray(labels, dtype=np.int64),
        expected_malware_probability=np.asarray(expected, dtype=np.float64),
        sample_ids=np.asarray(sample_ids),
    )
    _write_json(
        samples_dir / "parity_vectors.json",
        {
            "vectors": vectors,
            "expected_malware_probability": expected,
            "labels": labels,
            "sample_ids": [str(indices[i]) for i in range(num_samples)],
        },
    )
    _write_json(
        samples_dir / "index.json",
        {"num_samples": num_samples, "seed": seed, "samples": index_rows},
    )


def verify_onnx(onnx_path: Path, feature_dim: int) -> dict[str, Any]:
    import onnxruntime as ort

    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    inp_name = session.get_inputs()[0].name
    out_name = session.get_outputs()[0].name
    x = np.zeros((1, feature_dim), dtype=np.float32)
    out = session.run([out_name], {inp_name: x})[0]
    size_bytes = onnx_path.stat().st_size
    return {
        "input_name": inp_name,
        "output_name": out_name,
        "test_output_shape": list(out.shape),
        "test_output_value": float(out.ravel()[0]),
        "onnx_size_bytes": size_bytes,
        "onnx_size_kb": round(size_bytes / 1024, 2),
    }


def deploy_to_vigidroid(export_dir: Path, cfg) -> Path | None:
    repo_root = cfg.root.parent
    dest = repo_root / "vigidroid" / "app" / "src" / "main" / "assets" / "models" / cfg.model_id
    if not (repo_root / "vigidroid").is_dir():
        return None
    if dest.exists():
        shutil.rmtree(dest)
    shutil.copytree(export_dir, dest)
    return dest


def export_bundle(
    *,
    checkpoint: Path,
    out_dir: Path,
    config_path: Path | None = None,
    num_parity_samples: int = 10,
    seed: int = 42,
    skip_verify: bool = False,
    deploy_vigidroid: bool = False,
) -> Path:
    cfg = load_config(config_path)
    ensure_artifact_dirs(cfg)
    export_cfg = cfg.export
    opset = int(export_cfg.get("onnx_opset", 14))

    model, meta = load_export_model(checkpoint, cfg)
    out_dir = out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    onnx_path = out_dir / "model.onnx"
    export_onnx_model(model, onnx_path, feature_dim=meta["input_dim"], opset=opset)
    copy_feature_assets(cfg, out_dir)

    _write_json(out_dir / "thresholds.json", load_thresholds(cfg))

    pre_meta = load_preprocessing_meta(cfg.paths.processed)
    preprocessing_version = str(
        pre_meta.get("preprocessing_version") or _git_revision(cfg.root)
    )
    manifest = build_export_manifest(
        cfg,
        meta,
        opset=opset,
        preprocessing_version=preprocessing_version,
    )
    _write_json(out_dir / "export_manifest.json", manifest)

    val_path = feature_shard_path(cfg.paths.processed, "val")
    write_parity_samples(
        model,
        out_dir,
        val_shard_path=val_path,
        num_samples=num_parity_samples,
        seed=seed,
    )

    if not skip_verify:
        verify_info = verify_onnx(onnx_path, meta["input_dim"])
        manifest["onnx_runtime_check"] = verify_info
        _write_json(out_dir / "export_manifest.json", manifest)
        if verify_info["onnx_size_bytes"] > 20 * 1024:
            print(
                f"WARNING: ONNX size {verify_info['onnx_size_kb']} KB exceeds 20 KB target"
            )

    if deploy_vigidroid:
        dest = deploy_to_vigidroid(out_dir, cfg)
        if dest is not None:
            print(f"Deployed bundle → {dest}")

    return out_dir


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Export broadcast+MLDP hybrid ONNX bundle (P7).")
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=ROOT / "artifacts" / "checkpoints" / "best.pt",
    )
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--num-parity-samples", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--skip-verify", action="store_true")
    parser.add_argument(
        "--deploy-vigidroid",
        action="store_true",
        help="Copy bundle to vigidroid/app/src/main/assets/models/broadcast_mldp_hybrid/",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    cfg = load_config(args.config)
    out_dir = args.out_dir or cfg.paths.export
    num_parity = args.num_parity_samples
    if num_parity is None:
        num_parity = int(cfg.export.get("parity_num_samples", 10))

    bundle_dir = export_bundle(
        checkpoint=args.checkpoint.resolve(),
        out_dir=out_dir.resolve(),
        config_path=args.config,
        num_parity_samples=num_parity,
        seed=args.seed,
        skip_verify=args.skip_verify,
        deploy_vigidroid=args.deploy_vigidroid,
    )

    onnx_path = bundle_dir / "model.onnx"
    size_kb = onnx_path.stat().st_size / 1024
    print(f"Export bundle → {bundle_dir}")
    print(f"  model.onnx: {size_kb:.2f} KB")
    print(f"  features/: vocab + system_actions.json + feature_layout.json")
    print(f"  parity_samples/: {num_parity} samples")
    try:
        from src.thesis_archive import after_export

        after_export()
    except ImportError:
        pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
