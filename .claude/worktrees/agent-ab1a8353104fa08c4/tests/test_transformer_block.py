"""Round 144 — Transformer block."""

from __future__ import annotations

import pytest


class TestTransformer:
    def test_forward(self) -> None:
        pytest.importorskip("torch")
        import torch

        from naviertwin.core.neural.transformer_block import TransformerEncoderBlock

        blk = TransformerEncoderBlock(d_model=32, n_heads=4, d_ff=64)
        x = torch.randn(2, 10, 32)  # (batch, seq, d)
        y = blk(x)
        assert y.shape == x.shape

    def test_positional_encoding(self) -> None:
        pytest.importorskip("torch")

        from naviertwin.core.neural.transformer_block import positional_encoding

        pe = positional_encoding(seq_len=20, d_model=16)
        assert pe.shape == (20, 16)
        # 0번 위치의 홀수 인덱스 (cos(0)) = 1
        assert abs(float(pe[0, 1]) - 1.0) < 1e-6

    def test_train_step(self) -> None:
        pytest.importorskip("torch")
        import torch

        from naviertwin.core.neural.transformer_block import TransformerEncoderBlock

        blk = TransformerEncoderBlock(d_model=16, n_heads=2)
        opt = torch.optim.Adam(blk.parameters(), lr=1e-3)
        x = torch.randn(4, 8, 16)
        y = torch.randn(4, 8, 16)
        loss0 = ((blk(x) - y) ** 2).mean().item()
        for _ in range(50):
            opt.zero_grad()
            loss = ((blk(x) - y) ** 2).mean()
            loss.backward()
            opt.step()
        assert loss.item() < loss0
