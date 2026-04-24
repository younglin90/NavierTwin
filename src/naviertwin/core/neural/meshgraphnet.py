"""MeshGraphNet — Pfaff et al. 2020 (Encoder-Processor-Decoder).

간단 버전: node + edge MLP 인코딩 → message passing 라운드 → node 디코더.

Examples:
    >>> import torch
    >>> from naviertwin.core.neural.meshgraphnet import MeshGraphNet
    >>> mgn = MeshGraphNet(node_in=3, edge_in=2, hidden=16, out_dim=2, n_steps=2)
    >>> x = torch.randn(5, 3)
    >>> e = torch.randn(8, 2)
    >>> idx = torch.randint(0, 5, (2, 8))
    >>> y = mgn(x, e, idx)
    >>> y.shape
    torch.Size([5, 2])
"""

from __future__ import annotations

import torch
from torch import nn


def _mlp(d_in: int, d_hid: int, d_out: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(d_in, d_hid), nn.SiLU(),
        nn.Linear(d_hid, d_hid), nn.SiLU(),
        nn.Linear(d_hid, d_out),
    )


class MGNStep(nn.Module):
    def __init__(self, hidden: int) -> None:
        super().__init__()
        self.edge_mlp = _mlp(3 * hidden, hidden, hidden)
        self.node_mlp = _mlp(2 * hidden, hidden, hidden)

    def forward(self, h_n: torch.Tensor, h_e: torch.Tensor,
                edges: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        src, dst = edges[0], edges[1]
        e_in = torch.cat([h_e, h_n[src], h_n[dst]], dim=-1)
        h_e = h_e + self.edge_mlp(e_in)
        agg = torch.zeros_like(h_n).index_add(0, dst, h_e)
        n_in = torch.cat([h_n, agg], dim=-1)
        h_n = h_n + self.node_mlp(n_in)
        return h_n, h_e


class MeshGraphNet(nn.Module):
    def __init__(self, node_in: int, edge_in: int, hidden: int, out_dim: int,
                 n_steps: int = 3) -> None:
        super().__init__()
        self.node_enc = _mlp(node_in, hidden, hidden)
        self.edge_enc = _mlp(edge_in, hidden, hidden)
        self.steps = nn.ModuleList([MGNStep(hidden) for _ in range(n_steps)])
        self.dec = _mlp(hidden, hidden, out_dim)

    def forward(self, x: torch.Tensor, e_attr: torch.Tensor,
                edges: torch.Tensor) -> torch.Tensor:
        h_n = self.node_enc(x)
        h_e = self.edge_enc(e_attr)
        for s in self.steps:
            h_n, h_e = s(h_n, h_e, edges)
        return self.dec(h_n)


__all__ = ["MeshGraphNet"]
