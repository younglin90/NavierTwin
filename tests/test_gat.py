"""Round 281 — GAT layer."""

from __future__ import annotations

import pytest


class TestGAT:
    def test_forward_shape(self) -> None:
        torch = pytest.importorskip("torch")
        from naviertwin.core.neural.gat import GATLayer

        layer = GATLayer(in_dim=4, out_dim=8, heads=2)
        x = torch.randn(6, 4)
        edges = torch.tensor([[0, 1, 2, 3, 4, 5], [1, 2, 3, 4, 5, 0]])
        y = layer(x, edges)
        assert y.shape == (6, 16)

    def test_backward(self) -> None:
        torch = pytest.importorskip("torch")
        from naviertwin.core.neural.gat import GATLayer

        layer = GATLayer(in_dim=3, out_dim=5, heads=1)
        x = torch.randn(4, 3, requires_grad=True)
        edges = torch.tensor([[0, 1, 2], [1, 2, 3]])
        y = layer(x, edges)
        y.sum().backward()
        assert x.grad is not None
        assert x.grad.shape == x.shape
