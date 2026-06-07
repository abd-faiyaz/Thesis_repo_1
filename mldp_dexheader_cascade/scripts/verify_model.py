#!/usr/bin/env python3
"""P4 — verify Mode A/B models and paper sklearn baselines."""

from __future__ import annotations

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.config import load_config
from src.data.dataloaders import build_dataloaders_from_config
from src.models import (
    DeployedMlpHeaderRef,
    build_fused_mlp_from_config,
    build_mldp_logistic,
    build_mldp_stage1_from_config,
    count_parameters,
    estimate_fp32_bytes,
)
from src.models.mldp_logistic import MldpStage1TinyMlp
from src.training.svm_baseline import run_paper_baselines


def main() -> int:
    cfg = load_config()
    train_loader, _, _, feature_dims = build_dataloaders_from_config(cfg)

    s_dim = feature_dims["S"]
    h_dim = feature_dims["H"]
    d_dim = feature_dims["d"]

    mode_a = build_fused_mlp_from_config(cfg, d_dim)
    stage1 = build_mldp_stage1_from_config(cfg, s_dim)
    stage1_logistic = build_mldp_logistic(s_dim)
    stage1_tiny = MldpStage1TinyMlp(
        s_dim,
        hidden_dim=int(cfg.model.get("mode_b_stage1_mlp_hidden", 32)),
    )
    stage2 = DeployedMlpHeaderRef.from_config(cfg)

    dummy_x = torch.randn(1, d_dim)
    dummy_x_s = torch.randn(1, s_dim)
    dummy_h = torch.randn(1, h_dim)

    batch_x_s, batch_h, batch_x, _ = next(iter(train_loader))
    sample_x_s = batch_x_s[:4]
    sample_h = batch_h[:4]
    sample_x = batch_x[:4]

    with torch.no_grad():
        mode_a_logits = mode_a(dummy_x)
        stage1_logits = stage1(dummy_x_s)
        stage1_log_logits = stage1_logistic(dummy_x_s)
        stage1_tiny_logits = stage1_tiny(dummy_x_s)
        mode_a_batch = mode_a(sample_x)
        stage2_score = stage2.score(sample_h.numpy())

    mode_a_params = count_parameters(mode_a)
    mode_a_bytes = estimate_fp32_bytes(mode_a)
    stage1_params = count_parameters(stage1)

    print("P4 model check")
    print(f"  feature_dims: S={s_dim}  H={h_dim}  d={d_dim}")
    print(
        f"  Mode A fused MLP: out={tuple(mode_a_logits.shape)}  "
        f"params={mode_a_params:,}  fp32≈{mode_a_bytes/1024:.1f} KB"
    )
    print(
        f"  Mode B Stage 1 ({cfg.model.get('mode_b_stage1')}): "
        f"out={tuple(stage1_logits.shape)}  params={stage1_params:,}"
    )
    print(f"  Stage 1 logistic alt: out={tuple(stage1_log_logits.shape)}")
    print(f"  Stage 1 tiny MLP alt: out={tuple(stage1_tiny_logits.shape)}")
    print(f"  Mode B Stage 2 (deployed ONNX): scores={stage2_score.shape}  sample={stage2_score[:2]}")
    print(f"  Mode A batch: x={tuple(sample_x.shape)} → logits={tuple(mode_a_batch.shape)}")

    if mode_a_logits.shape != (1, 1):
        print(f"ERROR: Mode A logits shape {tuple(mode_a_logits.shape)} != (1, 1)")
        return 1
    if stage1_logits.shape != (1, 1):
        print(f"ERROR: Stage 1 logits shape {tuple(stage1_logits.shape)} != (1, 1)")
        return 1
    if stage2_score.shape[0] != 4:
        print(f"ERROR: Stage 2 batch score shape {stage2_score.shape}")
        return 1
    # Plan target: Mode A ONNX < 30 KB; fp32 weights are an upper bound.
    if mode_a_bytes > 40 * 1024:
        print(f"ERROR: Mode A param bytes {mode_a_bytes} exceed sanity bound")
        return 1

    print("\nP4 sklearn baseline smoke test (x_S only, train limit=500)...")
    baselines = run_paper_baselines(cfg, limit=500, save=False)
    for name, payload in baselines.items():
        print(
            f"  {name}: val F1={payload['val']['f1']:.4f}  "
            f"test F1={payload['test']['f1']:.4f}"
        )

    print("\nP4 exit criteria met.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
