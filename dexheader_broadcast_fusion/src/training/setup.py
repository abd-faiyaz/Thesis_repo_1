"""Optimizer and loss setup."""

from __future__ import annotations

import torch
import torch.nn as nn

from src.config import PipelineConfig


def build_training_objects(
    cfg: PipelineConfig,
    model: nn.Module,
    *,
    pos_weight: float,
) -> tuple[nn.Module, torch.optim.Optimizer, torch.device]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    pw = torch.tensor([pos_weight], dtype=torch.float32, device=device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pw)

    lr = float(cfg.training.get("learning_rate", 0.005))
    wd = float(cfg.training.get("weight_decay", 0.0001))
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    return criterion, optimizer, device


def build_fusion_model(
    cfg: PipelineConfig,
    *,
    dex_dim: int,
    receiver_dim: int,
) -> nn.Module:
    from src.models.fusion_net import FusionNet
    from src.models.warm_start import warm_start_header_tower

    model = FusionNet(
        dex_dim=dex_dim,
        receiver_dim=receiver_dim,
        header_hidden=int(cfg.model.get("header_hidden", 128)),
        receiver_embed_dim=int(cfg.model.get("receiver_embed_dim", 32)),
        fusion_hidden=int(cfg.model.get("fusion_hidden", 64)),
        fusion_head=str(cfg.model.get("fusion_head", "mlp")),
    )
    warm_started = False
    if bool(cfg.model.get("header_warm_start", True)):
        warm_started = warm_start_header_tower(
            model, cfg.paths.deployed_mlp_header_checkpoint
        )
    return model
