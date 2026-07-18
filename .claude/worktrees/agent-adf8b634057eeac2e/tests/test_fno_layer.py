"""Round 161 — FNO spectral layer."""

from __future__ import annotations

import pytest


class TestFNO:
    def test_spectral_conv_shape(self) -> None:
        pytest.importorskip("torch")
        import torch

        from naviertwin.core.neural.fno_layer import SpectralConv1d

        conv = SpectralConv1d(in_channels=4, out_channels=8, modes=6)
        x = torch.randn(2, 4, 32)
        y = conv(x)
        assert y.shape == (2, 8, 32)

    def test_fno_block(self) -> None:
        pytest.importorskip("torch")
        import torch

        from naviertwin.core.neural.fno_layer import FNOBlock1d

        blk = FNOBlock1d(channels=16, modes=8)
        x = torch.randn(4, 16, 64)
        y = blk(x)
        assert y.shape == x.shape

    def test_train_reduces_loss(self) -> None:
        pytest.importorskip("torch")
        import torch

        from naviertwin.core.neural.fno_layer import FNOBlock1d

        torch.manual_seed(0)
        blk = FNOBlock1d(channels=8, modes=4)
        x = torch.randn(4, 8, 32)
        y = torch.randn(4, 8, 32)
        opt = torch.optim.Adam(blk.parameters(), lr=1e-3)
        loss0 = ((blk(x) - y) ** 2).mean().item()
        for _ in range(50):
            opt.zero_grad()
            loss = ((blk(x) - y) ** 2).mean()
            loss.backward()
            opt.step()
        assert loss.item() < loss0
