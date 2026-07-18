"""Round 498 — NTK."""

from __future__ import annotations

import pytest


class TestNTK:
    def test_linear_ntk(self) -> None:
        torch = pytest.importorskip("torch")
        from naviertwin.utils.ntk import empirical_ntk

        # f(W, x) = W @ x; J wrt W is x ⊗ I
        W = torch.randn(2, 3, requires_grad=True)
        x1 = torch.randn(4, 3)
        x2 = torch.randn(5, 3)

        def fn(W, x):
            return x @ W.T

        K = empirical_ntk(fn, W, x1, x2)
        # K should be 2D, finite
        assert K.ndim == 2
        assert torch.isfinite(K).all()
