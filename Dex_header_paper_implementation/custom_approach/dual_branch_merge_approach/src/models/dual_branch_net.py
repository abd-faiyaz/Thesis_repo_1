"""Pattern B: MLP(H) + ASCNN(I) + fusion → malware logit."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
import torch.nn as nn

from src.models.ascnn_manifest import ASCNNManifest, build_ascnn_manifest_from_config
from src.models.fusion_head import FusionHead, build_fusion_head_from_config
from src.models.mlp_header import MLPHeaderBranch, build_mlp_header_branch_from_config

if TYPE_CHECKING:
    from src.config import PipelineConfig


class DualBranchNet(nn.Module):
    """
    Late fusion dual-branch classifier.

    forward(header, bow) -> logit (B, 1)
    predict_proba(header, bow) -> probability (B, 1)
    """

    def __init__(
        self,
        header_branch: MLPHeaderBranch,
        manifest_branch: ASCNNManifest,
        fusion_head: FusionHead,
    ) -> None:
        super().__init__()
        self.header_branch = header_branch
        self.manifest_branch = manifest_branch
        self.fusion_head = fusion_head

    def forward(
        self,
        header: torch.Tensor,
        bow: torch.Tensor,
    ) -> torch.Tensor:
        e_h = self.header_branch(header)
        e_i = self.manifest_branch(bow)
        fused = torch.cat([e_h, e_i], dim=-1)
        return self.fusion_head(fused)

    def predict_proba(
        self,
        header: torch.Tensor,
        bow: torch.Tensor,
    ) -> torch.Tensor:
        return torch.sigmoid(self.forward(header, bow))


def build_dual_branch_net(
    header_branch: MLPHeaderBranch | None = None,
    manifest_branch: ASCNNManifest | None = None,
    fusion_head: FusionHead | None = None,
    *,
    cfg: PipelineConfig | None = None,
) -> DualBranchNet:
    if cfg is None:
        if header_branch is None or manifest_branch is None or fusion_head is None:
            raise ValueError("Provide cfg or all three submodules")
        return DualBranchNet(header_branch, manifest_branch, fusion_head)

    return DualBranchNet(
        header_branch or build_mlp_header_branch_from_config(cfg),
        manifest_branch or build_ascnn_manifest_from_config(cfg),
        fusion_head or build_fusion_head_from_config(cfg),
    )


def build_dual_branch_net_from_config(cfg: PipelineConfig) -> DualBranchNet:
    return build_dual_branch_net(cfg=cfg)
