"""Round 284 — Laplace Neural Operator."""

from __future__ import annotations

import pytest


class TestLNO:
    def test_forward_shape(self) -> None:
        torch = pytest.importorskip("torch")
        from naviertwin.core.neural.lno import LNO1D

        m = LNO1D(channels=4, n_modes=8, n_layers=2)
        x = torch.randn(2, 4, 64)
        y = m(x)
        assert y.shape == x.shape

    def test_backward(self) -> None:
        torch = pytest.importorskip("torch")
        from naviertwin.core.neural.lno import LNO1D

        m = LNO1D(channels=2, n_modes=4, n_layers=1)
        x = torch.randn(1, 2, 32, requires_grad=True)
        y = m(x)
        y.sum().backward()
        assert x.grad is not None
