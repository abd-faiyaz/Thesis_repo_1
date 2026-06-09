#!/usr/bin/env python3
"""P7 — export two-input fusion ONNX deployment bundle."""

from __future__ import annotations

import argparse
import json
import random
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.config import ensure_artifact_dirs, load_config
from src.data.store import load_feature_shard, load_preprocessing_meta
from src.models.export_wrapper import FusionMalwareProbExport
from src.models.fusion_net import FusionNet
from src.training.checkpoint import load_best_checkpoint, restore_model_weights
from src.training.setup import build_fusion_model

ANDROID_ASSETS = "vigidroid/app/src/main/assets/models/dexheader_broadcast_fusion/"


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


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


def export_onnx_model(
    model: FusionMalwareProbExport,
    onnx_path: Path,
    *,
    dex_dim: int,
    receiver_dim: int,
    opset: int,
) -> None:
    onnx_path.parent.mkdir(parents=True, exist_ok=True)
    dummy_h = torch.zeros(1, dex_dim, dtype=torch.float32)
    dummy_r = torch.zeros(1, receiver_dim, dtype=torch.float32)
    torch.onnx.export(
        model,
        (dummy_h, dummy_r),
        str(onnx_path),
        export_params=True,
        opset_version=opset,
        do_constant_folding=True,
        dynamo=False,
        input_names=["dex_header", "receiver"],
        output_names=["malware_prob"],
        dynamic_axes={
            "dex_header": {0: "batch_size"},
            "receiver": {0: "batch_size"},
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
        "receiver_action_vocab.json",
        "feature_layout.json",
        "normalization_header.json",
        "system_actions.json",
    ):
        src = processed / name
        if not src.is_file():
            raise FileNotFoundError(f"Missing processed feature file: {src}")
        shutil.copy2(src, features_dir / name)


def write_parity_samples(
    cfg,
    export_dir: Path,
    model: FusionMalwareProbExport,
    *,
    num_samples: int,
) -> None:
    shard = load_feature_shard(cfg.paths.processed / "features_val.pt", split="val")
    n = min(num_samples, shard.H.shape[0])
    indices = random.sample(range(shard.H.shape[0]), n) if shard.H.shape[0] > n else list(range(n))

    samples_dir = export_dir / "parity_samples"
    if samples_dir.is_dir():
        shutil.rmtree(samples_dir)
    samples_dir.mkdir(parents=True)

    index: list[dict] = []
    model.eval()
    with torch.no_grad():
        for i, idx in enumerate(indices):
            H = shard.H[idx].numpy().astype(np.float32)
            R = shard.R[idx].numpy().astype(np.float32)
            prob = float(model(torch.from_numpy(H).unsqueeze(0), torch.from_numpy(R).unsqueeze(0)).item())
            sample_dir = samples_dir / f"sample_{i:03d}"
            sample_dir.mkdir()
            np.save(sample_dir / "H.npy", H)
            np.save(sample_dir / "R.npy", R)
            (sample_dir / "expected_prob.json").write_text(
                json.dumps({"malware_prob": prob}, indent=2) + "\n",
                encoding="utf-8",
            )
            index.append({"dir": sample_dir.name, "sha256": shard.sha256[idx]})
    (samples_dir / "index.json").write_text(json.dumps(index, indent=2) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="P7 ONNX export.")
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--deploy-vigidroid", action="store_true")
    args = parser.parse_args()

    cfg = load_config(args.config)
    ensure_artifact_dirs(cfg)

    payload = load_best_checkpoint(cfg.paths.checkpoints / "best.pt")
    receiver_dim = int(payload["receiver_dim"])
    dex_dim = int(payload.get("dex_dim", 104))
    model = build_fusion_model(cfg, dex_dim=dex_dim, receiver_dim=receiver_dim)
    restore_model_weights(model, payload)
    model.eval()
    export_model = FusionMalwareProbExport(model)
    export_model.eval()

    out_dir = cfg.paths.export
    out_dir.mkdir(parents=True, exist_ok=True)
    opset = int(cfg.export.get("onnx_opset", 14))
    export_onnx_model(
        export_model,
        out_dir / "model.onnx",
        dex_dim=dex_dim,
        receiver_dim=receiver_dim,
        opset=opset,
    )
    copy_feature_assets(cfg, out_dir)

    meta = load_preprocessing_meta(cfg.paths.processed)
    layout = json.loads((cfg.paths.processed / "feature_layout.json").read_text())
    r_size = int(layout["receiver"])
    d_r = int(layout.get("receiver_embed_dim", cfg.model.get("receiver_embed_dim", 32)))

    tuned_path = cfg.paths.metrics / "thresholds.json"
    if tuned_path.is_file():
        thresholds = json.loads(tuned_path.read_text())
    else:
        thresholds = {"default": 0.5, "tuned_val": 0.5}
    (out_dir / "thresholds.json").write_text(json.dumps(thresholds, indent=2) + "\n", encoding="utf-8")

    manifest = {
        "model_id": cfg.model_id,
        "domain": cfg.domain,
        "opset": opset,
        "inputs": [
            {"name": "dex_header", "shape": [1, dex_dim], "dtype": "float32"},
            {"name": "receiver", "shape": [1, r_size], "dtype": "float32"},
        ],
        "outputs": [{"name": "malware_prob", "dtype": "float32"}],
        "preprocessing_version": meta.get("preprocessing_version", _git_revision(cfg.root)),
        "multidex_mode": meta.get("multidex_mode", "sum"),
        "feature_extraction": {
            "apk_parts": ["classes*.dex(header)", "AndroidManifest.xml"],
            "fusion": "embedding_concat_then_fc",
            "branches": ["dex_header_104", "receiver_system_actions"],
            "dex_feature_dim": dex_dim,
            "dex_normalization": "per_byte_div255 -> multidex_sum -> bm1_corpus_minmax",
            "receiver_size_R": r_size,
            "receiver_embed_dim_dR": d_r,
            "receiver_system_actions_only": True,
        },
    }
    (out_dir / "export_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    write_parity_samples(
        cfg,
        out_dir,
        export_model,
        num_samples=int(cfg.export.get("parity_num_samples", 10)),
    )

    if args.deploy_vigidroid:
        repo_root = cfg.root.parent
        dest = repo_root / ANDROID_ASSETS
        if dest.is_dir():
            shutil.rmtree(dest)
        shutil.copytree(out_dir, dest)
        print(f"Deployed bundle → {dest}")

    print(f"Export complete → {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
