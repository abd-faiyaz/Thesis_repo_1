#!/usr/bin/env python3
"""P4 — verify deployment models and paper sklearn baselines."""

from __future__ import annotations

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.config import load_config
from src.data.dataloaders import build_dataloaders_from_config
from src.models import (
    build_deployment_model_from_config,
    build_logistic_head,
    build_tiny_mlp_from_config,
    count_parameters,
)
from src.training.svm_baseline import run_paper_baselines


def main() -> int:
    cfg = load_config()
    train_loader, _, _, feature_dim = build_dataloaders_from_config(cfg)

    tiny = build_tiny_mlp_from_config(cfg, feature_dim)
    logistic = build_logistic_head(feature_dim)
    deployment = build_deployment_model_from_config(cfg, feature_dim)

    dummy = torch.randn(1, feature_dim)
    batch_x, _ = next(iter(train_loader))

    with torch.no_grad():
        tiny_logits = tiny(dummy)
        logistic_logits = logistic(dummy)
        deploy_logits = deployment(dummy)
        batch_logits = tiny(batch_x[:4])

    tiny_params = count_parameters(tiny)
    logistic_params = count_parameters(logistic)

    print("P4 deployment model check")
    print(f"  feature_dim d: {feature_dim}")
    print(f"  deployment head: {cfg.classifier.get('deployment', 'tiny_mlp')}")
    print(f"  tiny_mlp:    dummy out {tuple(tiny_logits.shape)}  params={tiny_params:,}")
    print(f"  logistic:    dummy out {tuple(logistic_logits.shape)}  params={logistic_params:,}")
    print(f"  deployment:  dummy out {tuple(deploy_logits.shape)}")
    print(f"  tiny batch:  in={tuple(batch_x[:4].shape)} out={tuple(batch_logits.shape)}")

    if tiny_logits.shape != (1, 1):
        print(f"ERROR: expected tiny logits shape (1, 1), got {tuple(tiny_logits.shape)}")
        return 1
    if tiny_params > feature_dim * 64 * 4:
        print(f"ERROR: param count {tiny_params} exceeds sanity bound")
        return 1

    print("\nP4 sklearn baseline smoke test (train limit=500)...")
    baselines = run_paper_baselines(cfg, limit=500, save=False)
    for name, payload in baselines.items():
        val_f1 = payload["val"]["f1"]
        test_f1 = payload["test"]["f1"]
        print(f"  {name}: val F1={val_f1:.4f}  test F1={test_f1:.4f}")

    print("\nP4 exit criteria met.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
