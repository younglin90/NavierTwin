"""Graph message passing — 메쉬 그래프 기반 GNN 레이어.

Aggregation: mean/sum of neighbor features + self MLP.

Examples:
    >>> import torch  # doctest: +SKIP
"""

from __future__ import annotations


def _torch():
    try:
        import torch

        return torch
    except ImportError as exc:
        raise RuntimeError("torch 필요") from exc


def GraphConv(in_features: int, out_features: int, *, aggr: str = "mean"):
    """edge_index (2, E) 기반 message passing layer."""
    _torch()
    import torch
    import torch.nn as nn

    class _GC(nn.Module):
        def __init__(self):
            super().__init__()
            self.lin_self = nn.Linear(in_features, out_features)
            self.lin_msg = nn.Linear(in_features, out_features)
            self.aggr = aggr

        def forward(self, x, edge_index):
            """x: (N, F), edge_index: (2, E) long tensor (src, dst)."""
            N = x.size(0)
            src, dst = edge_index[0], edge_index[1]
            msg = self.lin_msg(x[src])  # (E, F_out)
            out = torch.zeros(N, msg.size(-1), device=x.device, dtype=msg.dtype)
            out.index_add_(0, dst, msg)
            if self.aggr == "mean":
                deg = torch.zeros(N, device=x.device, dtype=msg.dtype)
                ones = torch.ones_like(dst, dtype=msg.dtype)
                deg.index_add_(0, dst, ones)
                deg = deg.clamp(min=1.0)
                out = out / deg.unsqueeze(-1)
            elif self.aggr != "sum":
                raise ValueError(f"aggr ∈ mean/sum, got {aggr}")
            return self.lin_self(x) + out

    return _GC()


def build_edge_index_from_grid(
    nx: int, ny: int,
):
    """직사각 격자 → 4-neighbor edge_index (undirected)."""
    torch = _torch()
    nodes = torch.arange(nx * ny, dtype=torch.long).reshape(ny, nx)
    edge_blocks = []
    if nx > 1:
        left = nodes[:, :-1].reshape(-1)
        right = nodes[:, 1:].reshape(-1)
        edge_blocks.append(torch.stack((torch.cat((left, right)), torch.cat((right, left)))))
    if ny > 1:
        top = nodes[:-1, :].reshape(-1)
        bottom = nodes[1:, :].reshape(-1)
        edge_blocks.append(torch.stack((torch.cat((top, bottom)), torch.cat((bottom, top)))))
    if not edge_blocks:
        return torch.empty((2, 0), dtype=torch.long)
    return torch.cat(edge_blocks, dim=1)


__all__ = ["GraphConv", "build_edge_index_from_grid"]
