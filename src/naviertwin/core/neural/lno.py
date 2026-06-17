"""Laplace Neural Operator (LNO) — Cao et al. 2024.

핵심: spectral 변환을 Laplace transform 으로, learnable poles + residues.
간단 1D 버전: real-valued exponential basis weighting.

Examples:
    >>> import torch
    >>> from naviertwin.core.neural.lno import LNO1D
    >>> m = LNO1D(channels=4, n_modes=8, n_layers=2)
    >>> x = torch.randn(2, 4, 64)
    >>> y = m(x)
    >>> y.shape
    torch.Size([2, 4, 64])
"""

from __future__ import annotations

import torch
from torch import nn


class LNOBlock1D(nn.Module):
    def __init__(self, channels: int, n_modes: int) -> None:
        super().__init__()
        self.channels = channels
        self.n_modes = n_modes
        # learnable poles (real, stability-constrained) and residues
        self.log_neg_poles = nn.Parameter(torch.randn(n_modes) * 0.1)
        self.residues = nn.Parameter(torch.randn(channels, channels, n_modes) * 0.1)
        self.bias_w = nn.Conv1d(channels, channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, C, N)."""
        N = x.shape[-1]
        t = torch.linspace(0.0, 1.0, N, device=x.device)
        poles = torch.exp(self.log_neg_poles)
        basis = torch.exp(-poles[:, None] * t[None, :])  # (M, N)
        coef = (x @ basis.t()) / N  # (B, C, M)
        mixed = torch.einsum("oim,bim->bom", self.residues, coef)
        y = mixed @ basis
        return y + self.bias_w(x)


class LNO1D(nn.Module):
    def __init__(self, channels: int = 8, n_modes: int = 8,
                 n_layers: int = 2) -> None:
        super().__init__()
        self.layers = nn.ModuleList(
            map(lambda _: LNOBlock1D(channels, n_modes), range(n_layers)),
        )
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        layer_idx = 0
        while layer_idx < len(self.layers):
            x = self.act(self.layers[layer_idx](x))
            layer_idx += 1
        return x


__all__ = ["LNO1D", "LNOBlock1D"]
