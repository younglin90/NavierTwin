"""HyperNetwork — 입력 task code z → target net 의 weight 생성.

Examples:
    >>> import torch
    >>> from naviertwin.core.neural.hypernet import HyperNet
    >>> hn = HyperNet(z_dim=4, target_in=3, target_out=2, target_hidden=8)
    >>> z = torch.randn(2, 4)
    >>> x = torch.randn(2, 5, 3)
    >>> y = hn(z, x)
    >>> y.shape
    torch.Size([2, 5, 2])
"""

from __future__ import annotations

import torch
from torch import nn


class HyperNet(nn.Module):
    """z → linear-target weights {W1, b1, W2, b2}, 1-hidden MLP target."""

    def __init__(
        self, z_dim: int, target_in: int, target_out: int, target_hidden: int = 16,
    ) -> None:
        super().__init__()
        self.target_in = target_in
        self.target_out = target_out
        self.target_hidden = target_hidden
        n_params = (
            target_in * target_hidden + target_hidden
            + target_hidden * target_out + target_out
        )
        self.gen = nn.Sequential(
            nn.Linear(z_dim, 64), nn.SiLU(),
            nn.Linear(64, n_params),
        )

    def forward(self, z: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """z: (B, z_dim); x: (B, N, target_in) → (B, N, target_out)."""
        B = z.shape[0]
        params = self.gen(z)
        h = self.target_hidden
        ti, to = self.target_in, self.target_out
        idx = 0
        W1 = params[:, idx:idx + ti * h].reshape(B, ti, h)
        idx += ti * h
        b1 = params[:, idx:idx + h]
        idx += h
        W2 = params[:, idx:idx + h * to].reshape(B, h, to)
        idx += h * to
        b2 = params[:, idx:idx + to]
        # x @ W1 + b1
        h1 = torch.einsum("bni,bih->bnh", x, W1) + b1.unsqueeze(1)
        h1 = torch.tanh(h1)
        return torch.einsum("bnh,bho->bno", h1, W2) + b2.unsqueeze(1)


__all__ = ["HyperNet"]
