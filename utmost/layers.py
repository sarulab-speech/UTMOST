"""Layers shared by the WavLM and UTMOS22 encoders."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor, nn


def _weight_norm(v: Tensor, g: Tensor, dim: int) -> Tensor:
    """w = g * v / ||v||, with the norm taken over every dimension except `dim`."""
    fused = getattr(torch, "_weight_norm", None)
    if fused is not None:
        return fused(v, g, dim)
    norm = v.norm(p=2, dim=[d for d in range(v.dim()) if d != dim], keepdim=True)
    return v * (g / norm)


class WeightNormConv1d(nn.Module):
    """Conv1d whose weight is parameterized as weight_g (magnitude) and weight_v (direction)."""

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, padding: int, groups: int, dim: int = 2):
        super().__init__()
        self.padding, self.groups, self.dim = padding, groups, dim
        shape = (out_channels, in_channels // groups, kernel_size)
        g_shape = [1, 1, 1]
        g_shape[dim] = shape[dim]
        self.weight_g = nn.Parameter(torch.ones(g_shape))
        self.weight_v = nn.Parameter(torch.randn(shape) * 0.02)
        self.bias = nn.Parameter(torch.zeros(out_channels))

    def forward(self, x: Tensor) -> Tensor:
        weight = _weight_norm(self.weight_v, self.weight_g, self.dim)
        return F.conv1d(x, weight, self.bias, padding=self.padding, groups=self.groups)


def rename_weight_norm_keys(state: dict[str, Tensor]) -> dict[str, Tensor]:
    """Map `<conv>.parametrizations.weight.original0/1` (torch parametrizations) to `<conv>.weight_g/weight_v`."""
    renamed = {}
    for key, value in state.items():
        key = key.replace(".parametrizations.weight.original0", ".weight_g")
        key = key.replace(".parametrizations.weight.original1", ".weight_v")
        renamed[key] = value
    return renamed
