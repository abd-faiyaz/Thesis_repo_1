"""Two-tower embedding fusion: concat(z_H, z_R) → FC head → logits."""

from __future__ import annotations

import torch
import torch.nn as nn

from src.models.header_tower import HeaderTower
from src.models.receiver_tower import ReceiverTower


class FusionNet(nn.Module):
    def __init__(
        self,
        *,
        dex_dim: int = 104,
        receiver_dim: int,
        header_hidden: int = 128,
        receiver_embed_dim: int = 32,
        fusion_hidden: int = 64,
        fusion_head: str = "mlp",
    ) -> None:
        super().__init__()
        self.header_tower = HeaderTower(input_dim=dex_dim, hidden_dim=header_hidden)
        self.receiver_tower = ReceiverTower(receiver_dim, embed_dim=receiver_embed_dim)
        fused_dim = header_hidden + receiver_embed_dim
        if fusion_head == "logistic":
            self.fusion_head = nn.Linear(fused_dim, 1)
        else:
            self.fusion_head = nn.Sequential(
                nn.Linear(fused_dim, fusion_hidden),
                nn.ReLU(inplace=True),
                nn.Dropout(p=0.2),
                nn.Linear(fusion_hidden, 1),
            )
        self.fusion_head_type = fusion_head

    def forward(self, H: torch.Tensor, R: torch.Tensor) -> torch.Tensor:
        z_h = self.header_tower(H)
        z_r = self.receiver_tower(R)
        z = torch.cat([z_h, z_r], dim=-1)
        return self.fusion_head(z)
