"""P7 — export Mode A + Mode B ONNX deployment bundle."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn

from src.config import PipelineConfig, ensure_artifact_dirs, load_config
from src.data.store import feature_shard_path, load_feature_shard
from src.models import (
    DeployedMlpHeaderRef,
    build_fused_mlp_from_config,
    build_mldp_logistic,
)
from src.models.export_wrapper import MalwareProbExport, Stage1ProbExport
from src.models.mldp_logistic import MldpStage1TinyMlp
from src.training.checkpoint import load_checkpoint, restore_model_weights

_PACKAGE_ROOT = Path(__file__).resolve().parent.parent.parent

ANDROID_ASSETS = "vigidroid/app/src/main/assets/models/mldp_dexheader_cascade/"


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


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_preprocessing_version(processed_dir: Path, root: Path) -> str:
    meta_path = processed_dir / "preprocessing_meta.json"
    if meta_path.is_file():
        payload = json.loads(meta_path.read_text(encoding="utf-8"))
        version = payload.get("preprocessing_version")
        if version:
            return str(version)
    return _git_revision(root)


def load_mode_a_export_model(
    cfg: PipelineConfig,
    checkpoint_path: Path,
) -> tuple[MalwareProbExport, dict[str, Any]]:
    payload = load_checkpoint(checkpoint_path)
    input_dim = int(payload["d"])
    base = build_fused_mlp_from_config(cfg, input_dim)
    restore_model_weights(base, payload)
    base.eval()
    export_model = MalwareProbExport(base)
    export_model.eval()
    meta = {
        "checkpoint": str(checkpoint_path.resolve()),
        "input_dim": input_dim,
        "mode": "A",
        "config_hash": payload.get("config_hash"),
        "val_metrics": payload.get("val_metrics", {}),
        "feature_layout": payload.get("feature_layout", {}),
    }
    return export_model, meta


def load_stage1_export_model(
    cfg: PipelineConfig,
    checkpoint_path: Path,
) -> tuple[Stage1ProbExport, dict[str, Any]]:
    payload = load_checkpoint(checkpoint_path)
    head = str(payload.get("head", "logistic"))
    s_dim = int(payload["S_dim"])
    if head == "tiny_mlp":
        base = MldpStage1TinyMlp(
            s_dim,
            hidden_dim=int(cfg.model.get("mode_b_stage1_mlp_hidden", 32)),
        )
    else:
        base = build_mldp_logistic(s_dim)
    restore_model_weights(base, payload)
    base.eval()
    export_model = Stage1ProbExport(base)
    export_model.eval()
    meta = {
        "checkpoint": str(checkpoint_path.resolve()),
        "input_dim": s_dim,
        "mode": "B",
        "stage": 1,
        "head": head,
        "config_hash": payload.get("config_hash"),
        "val_metrics": payload.get("val_metrics", {}),
    }
    return export_model, meta


def export_onnx_model(
    model: nn.Module,
    onnx_path: Path,
    *,
    feature_dim: int,
    opset: int,
    input_name: str,
    output_name: str,
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
        input_names=[input_name],
        output_names=[output_name],
        dynamic_axes={
            input_name: {0: "batch_size"},
            output_name: {0: "batch_size"},
        },
    )
    data_sidecar = onnx_path.with_suffix(".onnx.data")
    if data_sidecar.is_file():
        data_sidecar.unlink()


def copy_feature_assets(cfg: PipelineConfig, out_dir: Path) -> None:
    features_dir = out_dir / "features"
    features_dir.mkdir(parents=True, exist_ok=True)
    processed = cfg.paths.processed
    for name in (
        "mldp_permission_vocab.json",
        "normalization_header.json",
        "feature_layout.json",
    ):
        src = processed / name
        if not src.is_file():
            raise FileNotFoundError(f"Missing processed feature file: {src}")
        shutil.copy2(src, features_dir / name)


def copy_stage2_onnx(cfg: PipelineConfig, dest_path: Path) -> dict[str, Any]:
    bundle = cfg.paths.deployed_mlp_header_bundle.resolve()
    src_onnx = bundle / "model.onnx"
    if not src_onnx.is_file():
        raise FileNotFoundError(f"Missing deployed Stage-2 ONNX: {src_onnx}")
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src_onnx, dest_path)
    src_hash = _sha256_file(src_onnx)
    dest_hash = _sha256_file(dest_path)
    if src_hash != dest_hash:
        raise RuntimeError("Stage-2 ONNX byte-copy verification failed")
    return {
        "source": str(src_onnx),
        "dest": str(dest_path.resolve()),
        "sha256": src_hash,
        "size_bytes": dest_path.stat().st_size,
    }


def load_thresholds(cfg: PipelineConfig) -> dict[str, Any]:
    tuned_path = cfg.paths.metrics / "thresholds.json"
    if tuned_path.is_file():
        return json.loads(tuned_path.read_text(encoding="utf-8"))
    default = float(cfg.evaluation.get("threshold", 0.5))
    return {
        "model_id": cfg.model_id,
        "mode_a": {"default": default, "tuned_val": default},
        "mode_b": {
            "stage1_t_low": 0.0,
            "stage1_t_high": 1.0,
        },
    }


def build_mode_a_manifest(
    cfg: PipelineConfig,
    meta: dict[str, Any],
    *,
    opset: int,
    preprocessing_version: str,
    layout: dict[str, Any],
) -> dict[str, Any]:
    d = int(layout["d"])
    s_size = int(layout["S"])
    h_size = int(layout["H"])
    dex_cfg = cfg.dex
    return {
        "model_id": cfg.model_id,
        "mode": "A",
        "domain": cfg.domain,
        "exported_at": _utc_now(),
        "opset": opset,
        "checkpoint": meta["checkpoint"],
        "config_hash": meta.get("config_hash"),
        "preprocessing_version": preprocessing_version,
        "multidex_mode": str(dex_cfg.get("multidex_mode", "sum")),
        "feature_extraction": {
            "apk_parts": ["AndroidManifest.xml", "classes*.dex(header)"],
            "fusion": "early_concat",
            "blocks": ["mldp_perms", "dex_header_104"],
            "mldp_size_S": s_size,
            "dex_feature_dim": h_size,
            "dex_normalization": "per_byte_div255 -> multidex_sum -> corpus_minmax",
        },
        "inputs": [
            {
                "name": "features",
                "shape": [1, d],
                "dtype": "float32",
                "description": "Early-fused [x_S || H], float32",
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


def build_mode_b_manifest(
    cfg: PipelineConfig,
    stage1_meta: dict[str, Any],
    stage2_copy: dict[str, Any],
    *,
    opset: int,
    preprocessing_version: str,
    layout: dict[str, Any],
) -> dict[str, Any]:
    s_size = int(layout["S"])
    h_size = int(layout["H"])
    dex_cfg = cfg.dex
    deployed_manifest = cfg.paths.deployed_mlp_header_bundle / "export_manifest.json"
    stage2_output = "malware_probability"
    if deployed_manifest.is_file():
        deployed = json.loads(deployed_manifest.read_text(encoding="utf-8"))
        outputs = deployed.get("outputs") or []
        if outputs:
            stage2_output = str(outputs[0].get("name", stage2_output))

    return {
        "model_id": cfg.model_id,
        "mode": "B",
        "domain": cfg.domain,
        "exported_at": _utc_now(),
        "opset": opset,
        "stage1_checkpoint": stage1_meta["checkpoint"],
        "stage1_head": stage1_meta.get("head"),
        "stage2_source": stage2_copy["source"],
        "stage2_sha256": stage2_copy["sha256"],
        "config_hash": stage1_meta.get("config_hash"),
        "preprocessing_version": preprocessing_version,
        "multidex_mode": str(dex_cfg.get("multidex_mode", "sum")),
        "feature_extraction": {
            "apk_parts": ["AndroidManifest.xml", "classes*.dex(header)"],
            "fusion": "cascade",
            "blocks": ["mldp_perms", "dex_header_104"],
            "mldp_size_S": s_size,
            "dex_feature_dim": h_size,
            "dex_normalization": "per_byte_div255 -> multidex_sum -> corpus_minmax",
            "stage2_normalization": "features/normalization_header.json (deployed mlp_header)",
        },
        "stage1": {
            "onnx_file": "stage1_mldp.onnx",
            "inputs": [
                {
                    "name": "features",
                    "shape": [1, s_size],
                    "dtype": "float32",
                    "description": "MLDP permission binary vector x_S",
                }
            ],
            "outputs": [
                {
                    "name": "stage1_prob",
                    "shape": [1, 1],
                    "dtype": "float32",
                    "description": "Stage-1 malware probability (sigmoid in ONNX graph)",
                }
            ],
        },
        "stage2": {
            "onnx_file": "stage2_mlp_header.onnx",
            "byte_copy": True,
            "inputs": [
                {
                    "name": "features",
                    "shape": [1, h_size],
                    "dtype": "float32",
                    "description": "Dex header vector H (deployed mlp_header normalization)",
                }
            ],
            "outputs": [
                {
                    "name": stage2_output,
                    "shape": [1, 1],
                    "dtype": "float32",
                    "description": "Stage-2 malware probability from deployed mlp_header",
                }
            ],
        },
        "android_assets_target": ANDROID_ASSETS,
    }


def write_parity_samples(
    cfg: PipelineConfig,
    out_dir: Path,
    *,
    mode_a_model: MalwareProbExport,
    stage1_model: Stage1ProbExport,
    stage2_ref: DeployedMlpHeaderRef,
    val_shard_path: Path,
    num_samples: int,
    seed: int,
) -> None:
    shard = load_feature_shard(val_shard_path, split="val")
    n = shard.x.shape[0]
    num_samples = min(num_samples, n)
    rng = np.random.default_rng(seed)
    indices = rng.choice(n, size=num_samples, replace=False)

    index_rows: list[dict[str, Any]] = []
    mode_a_model.eval()
    stage1_model.eval()

    with torch.no_grad():
        for i, idx in enumerate(indices):
            x_s = shard.x_s[int(idx)].float().unsqueeze(0)
            h = shard.h[int(idx)].float().unsqueeze(0)
            x = shard.x[int(idx)].float().unsqueeze(0)
            label = int(shard.y[int(idx)].item())

            mode_a_prob = float(mode_a_model(x).view(-1)[0].item())
            stage1_prob = float(stage1_model(x_s).view(-1)[0].item())
            stage2_prob = float(
                stage2_ref.score(h.numpy().astype(np.float32))[0]
            )

            sid = f"sample_{i:03d}"
            sample_dir = out_dir / "parity_samples" / sid
            sample_dir.mkdir(parents=True, exist_ok=True)
            np.save(sample_dir / "x_S.npy", x_s.numpy().astype(np.float32))
            np.save(sample_dir / "H.npy", h.numpy().astype(np.float32))
            np.save(sample_dir / "x.npy", x.numpy().astype(np.float32))
            _write_json(
                sample_dir / "expected_prob.json",
                {
                    "mode_a_malware_prob": mode_a_prob,
                    "stage1_prob": stage1_prob,
                    "stage2_prob": stage2_prob,
                },
            )
            index_rows.append(
                {
                    "sample_id": sid,
                    "index": int(idx),
                    "label": label,
                    "mode_a_malware_prob": mode_a_prob,
                    "stage1_prob": stage1_prob,
                    "stage2_prob": stage2_prob,
                    "dims": {"S": int(x_s.shape[-1]), "H": int(h.shape[-1]), "d": int(x.shape[-1])},
                }
            )

    _write_json(
        out_dir / "parity_samples" / "index.json",
        {"num_samples": num_samples, "seed": seed, "samples": index_rows},
    )


def verify_onnx_session(
    onnx_path: Path,
    feature_dim: int,
    *,
    input_name: str = "features",
    output_name: str | None = None,
) -> dict[str, Any]:
    import onnxruntime as ort

    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    inp_name = session.get_inputs()[0].name
    out_name = output_name or session.get_outputs()[0].name
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


def deploy_to_vigidroid(export_dir: Path, cfg: PipelineConfig) -> Path | None:
    repo_root = cfg.root.parent
    dest = repo_root / "vigidroid" / "app" / "src" / "main" / "assets" / "models" / cfg.model_id
    if not (repo_root / "vigidroid").is_dir():
        return None
    if dest.exists():
        shutil.rmtree(dest)
    shutil.copytree(export_dir, dest)
    return dest


def export_bundle(
    cfg: PipelineConfig,
    *,
    mode_a_checkpoint: Path | None = None,
    stage1_checkpoint: Path | None = None,
    out_dir: Path | None = None,
    num_parity_samples: int = 10,
    seed: int = 42,
    skip_verify: bool = False,
    deploy_vigidroid: bool = False,
) -> Path:
    ensure_artifact_dirs(cfg)
    export_cfg = cfg.export
    opset = int(export_cfg.get("onnx_opset", 14))

    mode_a_ckpt = mode_a_checkpoint or (cfg.paths.checkpoints / "mode_a_best.pt")
    stage1_ckpt = stage1_checkpoint or (cfg.paths.checkpoints / "stage1_best.pt")
    bundle_dir = (out_dir or cfg.paths.export).resolve()
    bundle_dir.mkdir(parents=True, exist_ok=True)

    layout = json.loads(
        (cfg.paths.processed / "feature_layout.json").read_text(encoding="utf-8")
    )
    preprocessing_version = load_preprocessing_version(cfg.paths.processed, cfg.root)
    thresholds = load_thresholds(cfg)

    mode_a_model, mode_a_meta = load_mode_a_export_model(cfg, mode_a_ckpt)
    stage1_model, stage1_meta = load_stage1_export_model(cfg, stage1_ckpt)
    stage2_ref = DeployedMlpHeaderRef.from_config(cfg)

    mode_a_dir = bundle_dir / "mode_a"
    mode_b_dir = bundle_dir / "mode_b"
    mode_a_onnx = mode_a_dir / "model.onnx"
    stage1_onnx = mode_b_dir / "stage1_mldp.onnx"
    stage2_onnx = mode_b_dir / "stage2_mlp_header.onnx"

    export_onnx_model(
        mode_a_model,
        mode_a_onnx,
        feature_dim=mode_a_meta["input_dim"],
        opset=opset,
        input_name="features",
        output_name="malware_prob",
    )
    export_onnx_model(
        stage1_model,
        stage1_onnx,
        feature_dim=stage1_meta["input_dim"],
        opset=opset,
        input_name="features",
        output_name="stage1_prob",
    )
    stage2_copy = copy_stage2_onnx(cfg, stage2_onnx)
    copy_feature_assets(cfg, bundle_dir)

    _write_json(bundle_dir / "thresholds.json", thresholds)
    _write_json(
        mode_a_dir / "thresholds.json",
        thresholds.get("mode_a", {"default": 0.5, "tuned_val": 0.5}),
    )
    _write_json(
        mode_b_dir / "thresholds.json",
        thresholds.get("mode_b", {"stage1_t_low": 0.0, "stage1_t_high": 1.0}),
    )

    mode_a_manifest = build_mode_a_manifest(
        cfg,
        mode_a_meta,
        opset=opset,
        preprocessing_version=preprocessing_version,
        layout=layout,
    )
    mode_b_manifest = build_mode_b_manifest(
        cfg,
        stage1_meta,
        stage2_copy,
        opset=opset,
        preprocessing_version=preprocessing_version,
        layout=layout,
    )

    val_path = feature_shard_path(cfg.paths.processed, "val")
    write_parity_samples(
        cfg,
        bundle_dir,
        mode_a_model=mode_a_model,
        stage1_model=stage1_model,
        stage2_ref=stage2_ref,
        val_shard_path=val_path,
        num_samples=num_parity_samples,
        seed=seed,
    )

    if not skip_verify:
        mode_a_manifest["onnx_runtime_check"] = verify_onnx_session(
            mode_a_onnx,
            mode_a_meta["input_dim"],
            output_name="malware_prob",
        )
        stage1_manifest_check = verify_onnx_session(
            stage1_onnx,
            stage1_meta["input_dim"],
            output_name="stage1_prob",
        )
        mode_b_manifest["stage1_onnx_runtime_check"] = stage1_manifest_check
        mode_b_manifest["stage2_onnx_size_bytes"] = stage2_copy["size_bytes"]

        if mode_a_manifest["onnx_runtime_check"]["onnx_size_bytes"] > 30 * 1024:
            print(
                "WARNING: Mode A ONNX size "
                f"{mode_a_manifest['onnx_runtime_check']['onnx_size_kb']} KB exceeds 30 KB target"
            )

    _write_json(mode_a_dir / "export_manifest.json", mode_a_manifest)
    _write_json(mode_b_dir / "export_manifest.json", mode_b_manifest)

    if deploy_vigidroid:
        dest = deploy_to_vigidroid(bundle_dir, cfg)
        if dest is not None:
            print(f"Deployed bundle → {dest}")

    return bundle_dir


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Export mldp_dexheader_cascade ONNX deployment bundle (P7)."
    )
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument(
        "--mode-a-checkpoint",
        type=Path,
        default=None,
        help="Default: artifacts/checkpoints/mode_a_best.pt",
    )
    parser.add_argument(
        "--stage1-checkpoint",
        type=Path,
        default=None,
        help="Default: artifacts/checkpoints/stage1_best.pt",
    )
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--num-parity-samples", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--skip-verify", action="store_true")
    parser.add_argument(
        "--deploy-vigidroid",
        action="store_true",
        help="Copy bundle to vigidroid/app/src/main/assets/models/mldp_dexheader_cascade/",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    if str(_PACKAGE_ROOT) not in sys.path:
        sys.path.insert(0, str(_PACKAGE_ROOT))

    args = build_arg_parser().parse_args(argv)
    cfg = load_config(args.config)
    num_parity = args.num_parity_samples
    if num_parity is None:
        num_parity = int(cfg.export.get("parity_num_samples", 10))

    bundle_dir = export_bundle(
        cfg,
        mode_a_checkpoint=args.mode_a_checkpoint.resolve() if args.mode_a_checkpoint else None,
        stage1_checkpoint=args.stage1_checkpoint.resolve() if args.stage1_checkpoint else None,
        out_dir=args.out_dir.resolve() if args.out_dir else None,
        num_parity_samples=num_parity,
        seed=args.seed,
        skip_verify=args.skip_verify,
        deploy_vigidroid=args.deploy_vigidroid,
    )

    mode_a_onnx = bundle_dir / "mode_a" / "model.onnx"
    stage1_onnx = bundle_dir / "mode_b" / "stage1_mldp.onnx"
    stage2_onnx = bundle_dir / "mode_b" / "stage2_mlp_header.onnx"
    print(f"Export bundle → {bundle_dir}")
    print(f"  mode_a/model.onnx: {mode_a_onnx.stat().st_size / 1024:.2f} KB")
    print(f"  mode_b/stage1_mldp.onnx: {stage1_onnx.stat().st_size / 1024:.2f} KB")
    print(f"  mode_b/stage2_mlp_header.onnx: {stage2_onnx.stat().st_size / 1024:.2f} KB (byte-copy)")
    print(f"  features/: vocab + normalization_header.json + feature_layout.json")
    print(f"  parity_samples/: {num_parity} samples")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
