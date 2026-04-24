"""Round 282 — MeshGraphNet."""

from __future__ import annotations

import pytest


class TestMGN:
    def test_forward_shape(self) -> None:
        torch = pytest.importorskip("torch")
        from naviertwin.core.neural.meshgraphnet import MeshGraphNet

        mgn = MeshGraphNet(node_in=4, edge_in=3, hidden=16, out_dim=2, n_steps=2)
        x = torch.randn(8, 4)
        e_attr = torch.randn(12, 3)
        edges = torch.randint(0, 8, (2, 12))
        y = mgn(x, e_attr, edges)
        assert y.shape == (8, 2)

    def test_backward(self) -> None:
        torch = pytest.importorskip("torch")
        from naviertwin.core.neural.meshgraphnet import MeshGraphNet

        mgn = MeshGraphNet(node_in=2, edge_in=1, hidden=8, out_dim=1, n_steps=1)
        x = torch.randn(5, 2, requires_grad=True)
        e_attr = torch.randn(6, 1)
        edges = torch.randint(0, 5, (2, 6))
        y = mgn(x, e_attr, edges)
        y.sum().backward()
        assert x.grad is not None
