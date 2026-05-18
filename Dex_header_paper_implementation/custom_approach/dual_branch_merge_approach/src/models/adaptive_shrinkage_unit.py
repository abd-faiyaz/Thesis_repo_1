"""Adaptive Shrinkage Unit (ASU): dynamic gating + soft threshold + Conv1d (paper Fig. 8)."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def soft_threshold(x: torch.Tensor, threshold: torch.Tensor) -> torch.Tensor:
    """
    Element-wise soft thresholding: sign(x) * relu(|x| - threshold).
    threshold broadcastable to x (typically B, C, 1).
    """
    return torch.sign(x) * F.relu(x.abs() - threshold)


class AdaptiveShrinkageUnit(nn.Module):
    """
    1-D adaptive shrinkage block for sparse BoW sequences.

    - Standard Conv1d with stride
    - Per-sample channel gate from global average pooling (dynamic scaling)
    - Per-sample soft threshold on conv output
    - BatchNorm1d + ReLU
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        *,
        kernel_size: int = 3,
        stride: int = 1,
    ) -> None:
        super().__init__()
        padding = kernel_size // 2
        self.conv = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
        )
        self.gate_fc = nn.Linear(in_channels, out_channels)
        self.threshold_fc = nn.Linear(out_channels, out_channels)
        self.bn = nn.BatchNorm1d(out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C_in, L)
        gap = x.mean(dim=-1)
        gate = torch.sigmoid(self.gate_fc(gap)).unsqueeze(-1)

        y = self.conv(x) * gate

        thresh_input = y.mean(dim=-1)
        threshold = F.softplus(self.threshold_fc(thresh_input)).unsqueeze(-1)
        y = soft_threshold(y, threshold)

        y = self.bn(y)
        return F.relu(y, inplace=True)
