"""Graph Attention Network (GAT) layer — Veličković et al. 2018.

Examples:
    >>> import torch
    >>> from naviertwin.core.neural.gat import GATLayer
    >>> layer = GATLayer(in_dim=4, out_dim=8, heads=2)
    >>> x = torch.randn(5, 4)
    >>> edges = torch.tensor([[0, 1], [1, 2], [2, 0]]).t()
    >>> y = layer(x, edges)
    >>> y.shape
    torch.Size([5, 16])
"""

from __future__ import annotations

import torch
from torch import nn


class GATLayer(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, heads: int = 1,
                 negative_slope: float = 0.2) -> None:
        super().__init__()
        self.heads = heads
        self.out_dim = out_dim
        self.W = nn.Linear(in_dim, heads * out_dim, bias=False)
        self.a_src = nn.Parameter(torch.randn(heads, out_dim) * 0.1)
        self.a_dst = nn.Parameter(torch.randn(heads, out_dim) * 0.1)
        self.act = nn.LeakyReLU(negative_slope)

    def forward(self, x: torch.Tensor, edges: torch.Tensor) -> torch.Tensor:
        """x: (n, in_dim); edges: (2, E) source→target."""
        n = x.shape[0]
        h = self.W(x).view(n, self.heads, self.out_dim)  # (n, H, F)
        src, dst = edges[0], edges[1]
        e_src = (h[src] * self.a_src).sum(-1)  # (E, H)
        e_dst = (h[dst] * self.a_dst).sum(-1)
        e = self.act(e_src + e_dst)  # (E, H)
        # softmax over incoming edges per dst
        e_max = torch.zeros(n, self.heads, device=x.device).scatter_reduce(
            0, dst.unsqueeze(-1).expand(-1, self.heads), e, reduce="amax",
            include_self=False,
        )
        e_exp = torch.exp(e - e_max[dst])
        denom = torch.zeros(n, self.heads, device=x.device).index_add(
            0, dst, e_exp,
        )
        alpha = e_exp / (denom[dst] + 1e-12)  # (E, H)
        msg = h[src] * alpha.unsqueeze(-1)  # (E, H, F)
        out = torch.zeros(n, self.heads, self.out_dim, device=x.device).index_add(
            0, dst, msg,
        )
        return out.reshape(n, self.heads * self.out_dim)


__all__ = ["GATLayer"]
