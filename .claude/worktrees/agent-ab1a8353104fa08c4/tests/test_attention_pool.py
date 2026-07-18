"""Round 202 — attention pooling."""

from __future__ import annotations

import pytest


class TestAttPool:
    def test_shape(self) -> None:
        pytest.importorskip("torch")
        import torch

        from naviertwin.core.neural.attention_pool import AttentionPool

        p = AttentionPool(d_model=16, n_heads=4)
        x = torch.randn(3, 20, 16)
        y = p(x)
        assert y.shape == (3, 16)

    def test_mean_max(self) -> None:
        pytest.importorskip("torch")
        import torch

        from naviertwin.core.neural.attention_pool import mean_max_pool

        x = torch.randn(2, 10, 8)
        y = mean_max_pool(x)
        assert y.shape == (2, 16)

    def test_gradient_flows(self) -> None:
        pytest.importorskip("torch")
        import torch

        from naviertwin.core.neural.attention_pool import AttentionPool

        p = AttentionPool(d_model=8, n_heads=2)
        x = torch.randn(2, 5, 8, requires_grad=True)
        y = p(x).sum()
        y.backward()
        assert x.grad is not None
