"""Round 146 — VAE."""

from __future__ import annotations

import pytest


class TestVAE:
    def test_forward(self) -> None:
        pytest.importorskip("torch")
        import torch

        from naviertwin.core.neural.vae import VAE, vae_loss

        torch.manual_seed(0)
        m = VAE(in_dim=10, latent_dim=4, hidden=16, n_layers=2)
        x = torch.randn(8, 10)
        x_hat, mu, logv = m(x)
        assert x_hat.shape == x.shape
        loss, parts = vae_loss(x_hat, x, mu, logv, beta=1.0)
        assert loss.requires_grad
        assert parts["rec"] >= 0
        assert parts["kl"] >= 0 or abs(parts["kl"]) < 1e6

    def test_training(self) -> None:
        pytest.importorskip("torch")
        import torch

        from naviertwin.core.neural.vae import VAE, vae_loss

        torch.manual_seed(0)
        m = VAE(in_dim=8, latent_dim=4, hidden=16)
        x = torch.randn(16, 8)
        opt = torch.optim.Adam(m.parameters(), lr=1e-3)
        x_hat, mu, logv = m(x)
        loss0, _ = vae_loss(x_hat, x, mu, logv, beta=0.1)
        loss0_val = float(loss0)
        for _ in range(100):
            opt.zero_grad()
            x_hat, mu, logv = m(x)
            loss, _ = vae_loss(x_hat, x, mu, logv, beta=0.1)
            loss.backward()
            opt.step()
        assert float(loss) < loss0_val
