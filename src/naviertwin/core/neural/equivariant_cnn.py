"""Equivariant CNN — SO(2) discrete rotation (4 회전 group C4).

각 group element 별로 동일한 conv kernel을 회전시켜 적용 후 평균 풀링 (group avg).
간단 group-equivariant 2D conv.

Examples:
    >>> import torch
    >>> from naviertwin.core.neural.equivariant_cnn import C4Conv2d
    >>> conv = C4Conv2d(1, 4, kernel_size=3)
    >>> x = torch.randn(1, 1, 16, 16)
    >>> y = conv(x)
    >>> y.shape
    torch.Size([1, 4, 16, 16])
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


class C4Conv2d(nn.Module):
    """4-fold rotation equivariant Conv2d (group avg over rotations)."""

    def __init__(self, in_ch: int, out_ch: int, kernel_size: int = 3) -> None:
        super().__init__()
        self.in_ch = in_ch
        self.out_ch = out_ch
        self.kernel_size = kernel_size
        self.weight = nn.Parameter(
            torch.randn(out_ch, in_ch, kernel_size, kernel_size) * 0.1,
        )
        self.bias = nn.Parameter(torch.zeros(out_ch))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        padding = self.kernel_size // 2
        outs = (
            F.conv2d(x, self.weight, bias=self.bias, padding=padding),
            F.conv2d(x, torch.rot90(self.weight, 1, dims=(-2, -1)), bias=self.bias, padding=padding),
            F.conv2d(x, torch.rot90(self.weight, 2, dims=(-2, -1)), bias=self.bias, padding=padding),
            F.conv2d(x, torch.rot90(self.weight, 3, dims=(-2, -1)), bias=self.bias, padding=padding),
        )
        return torch.stack(outs, dim=0).mean(dim=0)


__all__ = ["C4Conv2d"]
