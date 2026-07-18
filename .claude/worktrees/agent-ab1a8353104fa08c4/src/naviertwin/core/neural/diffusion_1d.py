"""Score-based diffusion 1D — Song & Ermon 2020 / DDPM 풍 (간단 forward+score net).

Examples:
    >>> import torch
    >>> from naviertwin.core.neural.diffusion_1d import (
    ...     ScoreNet1D, forward_diffusion,
    ... )
    >>> sn = ScoreNet1D(dim=8, hidden=32)
    >>> x = torch.randn(4, 8)
    >>> t = torch.rand(4)
    >>> s = sn(x, t)
    >>> s.shape
    torch.Size([4, 8])
"""

from __future__ import annotations

import torch
from torch import nn


def forward_diffusion(
    x0: torch.Tensor, t: torch.Tensor, beta_max: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """VE SDE 풍: x_t = x_0 + σ(t) ε, σ(t) = √(t·beta_max)."""
    sigma = torch.sqrt(t * beta_max).unsqueeze(-1)
    eps = torch.randn_like(x0)
    return x0 + sigma * eps, eps


class ScoreNet1D(nn.Module):
    def __init__(self, dim: int, hidden: int = 64) -> None:
        super().__init__()
        self.t_embed = nn.Sequential(nn.Linear(1, hidden), nn.SiLU())
        self.net = nn.Sequential(
            nn.Linear(dim + hidden, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden), nn.SiLU(),
            nn.Linear(hidden, dim),
        )

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        te = self.t_embed(t.unsqueeze(-1))
        return self.net(torch.cat([x, te], dim=-1))


__all__ = ["ScoreNet1D", "forward_diffusion"]
