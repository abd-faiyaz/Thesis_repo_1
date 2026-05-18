"""Pattern B model components (Phase 4)."""

from src.models.adaptive_shrinkage_unit import AdaptiveShrinkageUnit, soft_threshold
from src.models.ascnn_manifest import ASCNNManifest, build_ascnn_manifest, build_ascnn_manifest_from_config
from src.models.dual_branch_net import DualBranchNet, build_dual_branch_net, build_dual_branch_net_from_config
from src.models.fusion_head import FusionHead, build_fusion_head, build_fusion_head_from_config
from src.models.mlp_header import MLPHeaderBranch, build_mlp_header_branch, build_mlp_header_branch_from_config

__all__ = [
    "AdaptiveShrinkageUnit",
    "soft_threshold",
    "MLPHeaderBranch",
    "build_mlp_header_branch",
    "build_mlp_header_branch_from_config",
    "ASCNNManifest",
    "build_ascnn_manifest",
    "build_ascnn_manifest_from_config",
    "FusionHead",
    "build_fusion_head",
    "build_fusion_head_from_config",
    "DualBranchNet",
    "build_dual_branch_net",
    "build_dual_branch_net_from_config",
]
