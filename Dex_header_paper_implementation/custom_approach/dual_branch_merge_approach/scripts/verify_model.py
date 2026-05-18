#!/usr/bin/env python3
"""Phase 4: verify DualBranchNet forward pass with dummy or dataloader batch."""

from __future__ import annotations

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.config import load_config
from src.constants import DEX_HEADER_FEATURE_DIM
from src.data.dataloaders import build_dataloaders_from_config
from src.models.dual_branch_net import build_dual_branch_net_from_config


def main() -> int:
    cfg = load_config()
    model = build_dual_branch_net_from_config(cfg)
    model.eval()

    train_manifest = cfg.paths.manifest_train
    val_manifest = cfg.paths.manifest_val

    if train_manifest.is_file() and val_manifest.is_file():
        print(f"Using DataLoader from manifests")
        train_loader, _, header_dim, bow_dim = build_dataloaders_from_config(cfg)
        header, bow, _ = next(iter(train_loader))
    else:
        print("Manifests not found; using random dummy batch")
        batch_size = int(cfg.data.get("batch_size", 16))
        bow_dim = int(cfg.model.get("bow_padded_len", 4381))
        header_dim = DEX_HEADER_FEATURE_DIM
        header = torch.randn(batch_size, header_dim)
        bow = torch.randn(batch_size, bow_dim)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    header = header.to(device)
    bow = bow.to(device)

    with torch.no_grad():
        logits = model(header, bow)
        probs = model.predict_proba(header, bow)

    n_params = sum(p.numel() for p in model.parameters())

    print("Phase 4 DualBranchNet check")
    print(f"  device:     {device}")
    print(f"  header in:  {tuple(header.shape)}")
    print(f"  bow in:     {tuple(bow.shape)}")
    print(f"  logit out:  {tuple(logits.shape)}")
    print(f"  prob out:   {tuple(probs.shape)}")
    print(f"  prob range: [{probs.min():.4f}, {probs.max():.4f}]")
    print(f"  parameters: {n_params:,}")
    print("\nPhase 4 verified.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
