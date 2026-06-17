"""Conditional FNO 1D — parameter-conditioned spectral conv (FiLM 풍).

Examples:
    >>> import torch
    >>> from naviertwin.core.neural.cfno import CFNO1D
    >>> m = CFNO1D(in_ch=2, out_ch=2, modes=8, width=16, p_dim=3, n_layers=2)
    >>> x = torch.randn(2, 2, 64)
    >>> p = torch.randn(2, 3)
    >>> y = m(x, p)
    >>> y.shape
    torch.Size([2, 2, 64])
"""

from __future__ import annotations

import torch
from torch import nn


class _SpectralConv1d(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, modes: int) -> None:
        super().__init__()
        self.modes = modes
        scale = 1.0 / (in_ch * out_ch)
        self.weight = nn.Parameter(
            scale * torch.randn(in_ch, out_ch, modes, dtype=torch.cfloat),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, N = x.shape
        x_ft = torch.fft.rfft(x, dim=-1)
        out_ft = torch.zeros(
            B, self.weight.shape[1], x_ft.shape[-1], dtype=torch.cfloat,
            device=x.device,
        )
        m = min(self.modes, x_ft.shape[-1])
        out_ft[:, :, :m] = torch.einsum("bim,iom->bom", x_ft[:, :, :m], self.weight[:, :, :m])
        return torch.fft.irfft(out_ft, n=N, dim=-1)


class CFNOBlock(nn.Module):
    def __init__(self, ch: int, modes: int, p_dim: int) -> None:
        super().__init__()
        self.spec = _SpectralConv1d(ch, ch, modes)
        self.w = nn.Conv1d(ch, ch, 1)
        self.film = nn.Linear(p_dim, 2 * ch)

    def forward(self, x: torch.Tensor, p: torch.Tensor) -> torch.Tensor:
        gb = self.film(p)  # (B, 2C)
        g, b = gb.chunk(2, dim=-1)
        y = self.spec(x) + self.w(x)
        y = y * (1 + g.unsqueeze(-1)) + b.unsqueeze(-1)
        return torch.relu(y)


class CFNO1D(nn.Module):
    def __init__(
        self, in_ch: int = 1, out_ch: int = 1, modes: int = 16,
        width: int = 32, p_dim: int = 3, n_layers: int = 4,
    ) -> None:
        super().__init__()
        self.lift = nn.Conv1d(in_ch, width, 1)
        self.blocks = nn.ModuleList(
            map(lambda _: CFNOBlock(width, modes, p_dim), range(n_layers)),
        )
        self.proj = nn.Conv1d(width, out_ch, 1)

    def forward(self, x: torch.Tensor, p: torch.Tensor) -> torch.Tensor:
        x = self.lift(x)
        block_idx = 0
        while block_idx < len(self.blocks):
            x = self.blocks[block_idx](x, p)
            block_idx += 1
        return self.proj(x)


__all__ = ["CFNO1D"]
