"""Round 174 — GNN layer."""

from __future__ import annotations

import pytest


class TestGNN:
    def test_forward(self) -> None:
        pytest.importorskip("torch")
        import torch

        from naviertwin.core.neural.gnn_layer import (
            GraphConv,
            build_edge_index_from_grid,
        )

        nx, ny = 4, 3
        N = nx * ny
        ei = build_edge_index_from_grid(nx, ny)
        x = torch.randn(N, 8)
        gc = GraphConv(8, 16, aggr="mean")
        y = gc(x, ei)
        assert y.shape == (N, 16)

    def test_sum_vs_mean(self) -> None:
        pytest.importorskip("torch")
        import torch

        from naviertwin.core.neural.gnn_layer import GraphConv

        ei = torch.tensor([[0, 1, 2], [1, 2, 0]], dtype=torch.long)
        x = torch.randn(3, 4)
        gc_s = GraphConv(4, 4, aggr="sum")
        gc_m = GraphConv(4, 4, aggr="mean")
        assert gc_s(x, ei).shape == gc_m(x, ei).shape

    def test_train(self) -> None:
        pytest.importorskip("torch")
        import torch

        from naviertwin.core.neural.gnn_layer import (
            GraphConv,
            build_edge_index_from_grid,
        )

        torch.manual_seed(0)
        ei = build_edge_index_from_grid(4, 4)
        N = 16
        gc = GraphConv(3, 3)
        x = torch.randn(N, 3)
        y = torch.randn(N, 3)
        opt = torch.optim.Adam(gc.parameters(), lr=1e-2)
        loss0 = ((gc(x, ei) - y) ** 2).mean().item()
        for _ in range(80):
            opt.zero_grad()
            loss = ((gc(x, ei) - y) ** 2).mean()
            loss.backward()
            opt.step()
        assert loss.item() < loss0
