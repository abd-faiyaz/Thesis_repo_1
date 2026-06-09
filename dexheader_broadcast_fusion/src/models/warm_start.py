"""Load deployed BM1 trunk weights into FusionNet.header_tower."""

from __future__ import annotations

from pathlib import Path

import torch

from src.models.fusion_net import FusionNet


def _load_mlp_header_state(checkpoint_path: Path) -> dict[str, torch.Tensor]:
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if isinstance(payload, dict) and "model_state_dict" in payload:
        return payload["model_state_dict"]
    if isinstance(payload, dict) and "model_state" in payload:
        return payload["model_state"]
    raise ValueError(f"Unrecognized checkpoint format: {checkpoint_path}")


def warm_start_header_tower(model: FusionNet, checkpoint_path: Path) -> bool:
    """Copy block1/block2 from deployed MLP(H); skip classification head."""
    if not checkpoint_path.is_file():
        print(f"WARN: header warm-start checkpoint missing: {checkpoint_path}")
        return False

    state = _load_mlp_header_state(checkpoint_path)
    subset = {
        k: v for k, v in state.items() if k.startswith("block1.") or k.startswith("block2.")
    }
    if not subset:
        print(f"WARN: no block1/block2 keys in checkpoint: {checkpoint_path}")
        return False

    model.header_tower.load_state_dict(subset, strict=True)
    print(f"Header warm-start loaded {len(subset)} tensors from {checkpoint_path}")
    return True
