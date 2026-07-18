"""Round 286 — HyperNetwork."""

from __future__ import annotations

import pytest


class TestHyperNet:
    def test_forward_shape(self) -> None:
        torch = pytest.importorskip("torch")
        from naviertwin.core.neural.hypernet import HyperNet

        hn = HyperNet(z_dim=4, target_in=3, target_out=2, target_hidden=8)
        z = torch.randn(2, 4)
        x = torch.randn(2, 5, 3)
        y = hn(z, x)
        assert y.shape == (2, 5, 2)

    def test_different_z_different_outputs(self) -> None:
        torch = pytest.importorskip("torch")
        from naviertwin.core.neural.hypernet import HyperNet

        hn = HyperNet(z_dim=2, target_in=1, target_out=1, target_hidden=4)
        z1 = torch.zeros(1, 2)
        z2 = torch.ones(1, 2)
        x = torch.linspace(0, 1, 10).reshape(1, 10, 1)
        y1 = hn(z1, x)
        y2 = hn(z2, x)
        assert not torch.allclose(y1, y2)
