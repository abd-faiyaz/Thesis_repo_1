#!/usr/bin/env python3
"""Phase 4: verify MLP(H) forward pass with config + feature dim."""

from __future__ import annotations

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.config import load_config
from src.data.dataloaders import build_dataloaders_from_bundle, resolve_processed_path
from src.data.store import load_processed_bundle
from src.features.dex_header import FEATURE_DIM
from src.models.mlp_header import build_mlp_header_from_config


def _synthetic_bundle(n: int = 16):
    from src.data.store import ProcessedBundle

    return ProcessedBundle(
        features=torch.rand(n, FEATURE_DIM),
        labels=torch.randint(0, 2, (n,)).float(),
        paths=[f"synthetic_{i}.apk" for i in range(n)],
        feature_dim=FEATURE_DIM,
        source_path=Path("synthetic"),
    )


def main() -> int:
    cfg = load_config()
    processed_path = resolve_processed_path(cfg)

    if processed_path.is_file():
        bundle = load_processed_bundle(processed_path)
        feature_dim = bundle.feature_dim
    else:
        bundle = _synthetic_bundle()
        feature_dim = FEATURE_DIM

    model = build_mlp_header_from_config(cfg, input_dim=feature_dim)
    train_loader, _, _ = build_dataloaders_from_bundle(
        bundle,
        batch_size=int(cfg.data.get("batch_size", 16)),
        num_workers=0,
    )

    batch_x, _ = next(iter(train_loader))
    with torch.no_grad():
        probs = model(batch_x)

    n_params = sum(p.numel() for p in model.parameters())

    print("Phase 4 MLP(H) check")
    print(f"  input_dim:  {model.input_dim}")
    print(f"  hidden_dim: {model.hidden_dim}")
    print(f"  batch in:   {tuple(batch_x.shape)}")
    print(f"  batch out:  {tuple(probs.shape)}")
    print(f"  prob range: [{probs.min():.4f}, {probs.max():.4f}]")
    print(f"  parameters: {n_params:,}")
    print("\nPhase 4 verified.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
