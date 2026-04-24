"""Round 145 — Conv Autoencoder."""

from __future__ import annotations

import pytest


class TestConvAE:
    def test_shapes(self) -> None:
        pytest.importorskip("torch")
        import torch

        from naviertwin.core.neural.conv_ae import ConvAutoencoder

        ae = ConvAutoencoder(in_channels=2, latent_dim=16, input_size=32)
        x = torch.randn(4, 2, 32, 32)
        z = ae.encode(x)
        x_hat = ae.decode(z)
        assert z.shape == (4, 16)
        assert x_hat.shape == x.shape

    def test_train_reduces_loss(self) -> None:
        pytest.importorskip("torch")
        import torch

        from naviertwin.core.neural.conv_ae import ConvAutoencoder

        torch.manual_seed(0)
        ae = ConvAutoencoder(in_channels=1, latent_dim=8, input_size=16, base=8)
        x = torch.randn(8, 1, 16, 16)
        opt = torch.optim.Adam(ae.parameters(), lr=1e-3)
        loss0 = ((ae(x) - x) ** 2).mean().item()
        for _ in range(40):
            opt.zero_grad()
            loss = ((ae(x) - x) ** 2).mean()
            loss.backward()
            opt.step()
        assert loss.item() < loss0

    def test_invalid_size(self) -> None:
        pytest.importorskip("torch")
        from naviertwin.core.neural.conv_ae import ConvAutoencoder

        with pytest.raises(ValueError):
            ConvAutoencoder(input_size=15)
