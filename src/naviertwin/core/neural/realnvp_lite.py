"""RealNVP-lite — Dinh et al. 2017, affine coupling layers (간단 버전).

Examples:
    >>> import torch
    >>> from naviertwin.core.neural.realnvp_lite import RealNVPLite
    >>> flow = RealNVPLite(dim=4, hidden=16, n_layers=3)
    >>> x = torch.randn(8, 4)
    >>> z, logdet = flow(x)
    >>> z.shape, logdet.shape
    (torch.Size([8, 4]), torch.Size([8]))
"""

from __future__ import annotations

import torch
from torch import nn


class AffineCoupling(nn.Module):
    def __init__(self, dim: int, hidden: int, mask: torch.Tensor) -> None:
        super().__init__()
        self.register_buffer("mask", mask)
        self.s = nn.Sequential(
            nn.Linear(dim, hidden), nn.SiLU(),
            nn.Linear(hidden, dim), nn.Tanh(),
        )
        self.t = nn.Sequential(
            nn.Linear(dim, hidden), nn.SiLU(),
            nn.Linear(hidden, dim),
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x_m = x * self.mask
        s = self.s(x_m) * (1 - self.mask)
        t = self.t(x_m) * (1 - self.mask)
        z = x_m + (1 - self.mask) * (x * torch.exp(s) + t)
        logdet = s.sum(dim=-1)
        return z, logdet

    def inverse(self, z: torch.Tensor) -> torch.Tensor:
        z_m = z * self.mask
        s = self.s(z_m) * (1 - self.mask)
        t = self.t(z_m) * (1 - self.mask)
        x = z_m + (1 - self.mask) * (z - t) * torch.exp(-s)
        return x


class RealNVPLite(nn.Module):
    def __init__(self, dim: int = 4, hidden: int = 32, n_layers: int = 4) -> None:
        super().__init__()
        masks = []
        for i in range(n_layers):
            m = torch.zeros(dim)
            m[i % 2::2] = 1.0
            masks.append(m)
        self.layers = nn.ModuleList(
            [AffineCoupling(dim, hidden, m) for m in masks],
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        logdet = torch.zeros(x.shape[0], device=x.device)
        for layer in self.layers:
            x, ld = layer(x)
            logdet = logdet + ld
        return x, logdet

    def inverse(self, z: torch.Tensor) -> torch.Tensor:
        for layer in reversed(self.layers):
            z = layer.inverse(z)
        return z


__all__ = ["RealNVPLite", "AffineCoupling"]
