"""Round 283 — Operator Transformer block."""

from __future__ import annotations

import pytest


class TestOpTransformer:
    def test_self_attention_shape(self) -> None:
        torch = pytest.importorskip("torch")
        from naviertwin.core.neural.op_transformer import OpTransformerBlock

        blk = OpTransformerBlock(d_model=16, heads=4)
        x = torch.randn(2, 32, 16)
        y = blk(x)
        assert y.shape == x.shape

    def test_cross_attention(self) -> None:
        torch = pytest.importorskip("torch")
        from naviertwin.core.neural.op_transformer import OpTransformerBlock

        blk = OpTransformerBlock(d_model=8, heads=2)
        q = torch.randn(1, 10, 8)
        ctx = torch.randn(1, 20, 8)
        y = blk(q, context=ctx)
        assert y.shape == q.shape
