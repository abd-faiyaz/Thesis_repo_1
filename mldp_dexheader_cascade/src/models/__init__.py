"""Mode A fused MLP, Mode B Stage-1 heads, deployed MLP(H) reference."""

from src.models.factory import (
    build_mode_a_from_config,
    build_mode_b_stage1_from_config,
    build_mode_b_stage1_logistic,
    count_parameters,
    estimate_fp32_bytes,
)
from src.models.fused_mlp import FusedMlp, build_fused_mlp, build_fused_mlp_from_config
from src.models.mldp_logistic import (
    MldpLogistic,
    MldpStage1TinyMlp,
    build_mldp_logistic,
    build_mldp_stage1_from_config,
)
from src.models.mlp_header_ref import DeployedMlpHeaderRef

__all__ = [
    "DeployedMlpHeaderRef",
    "FusedMlp",
    "MldpLogistic",
    "MldpStage1TinyMlp",
    "build_fused_mlp",
    "build_fused_mlp_from_config",
    "build_mldp_logistic",
    "build_mldp_stage1_from_config",
    "build_mode_a_from_config",
    "build_mode_b_stage1_from_config",
    "build_mode_b_stage1_logistic",
    "count_parameters",
    "estimate_fp32_bytes",
]
