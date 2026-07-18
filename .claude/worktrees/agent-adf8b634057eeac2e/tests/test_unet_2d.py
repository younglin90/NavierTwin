"""Round 201 — U-Net 2D."""

from __future__ import annotations

import pytest


class TestUNet:
    def test_forward_shape(self) -> None:
        pytest.importorskip("torch")
        import torch

        from naviertwin.core.neural.unet_2d import UNet2D

        m = UNet2D(in_channels=2, out_channels=1, base=8)
        x = torch.randn(2, 2, 32, 32)
        y = m(x)
        assert y.shape == (2, 1, 32, 32)

    def test_train(self) -> None:
        pytest.importorskip("torch")
        import torch

        from naviertwin.core.neural.unet_2d import UNet2D

        torch.manual_seed(0)
        m = UNet2D(in_channels=1, out_channels=1, base=4)
        x = torch.randn(2, 1, 16, 16)
        y = torch.randn(2, 1, 16, 16)
        opt = torch.optim.Adam(m.parameters(), lr=1e-3)
        l0 = ((m(x) - y) ** 2).mean().item()
        for _ in range(40):
            opt.zero_grad()
            loss = ((m(x) - y) ** 2).mean()
            loss.backward()
            opt.step()
        assert loss.item() < l0
