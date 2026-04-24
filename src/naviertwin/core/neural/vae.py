"""Variational Autoencoder — MLP 기반 경량 VAE.

Examples:
    >>> import torch  # doctest: +SKIP
    >>> from naviertwin.core.neural.vae import VAE  # doctest: +SKIP
"""

from __future__ import annotations


def _torch():
    try:
        import torch

        return torch
    except ImportError as exc:
        raise RuntimeError("torch 필요") from exc


def VAE(
    in_dim: int, latent_dim: int = 8, hidden: int = 64, n_layers: int = 2,
):
    """MLP VAE (encoder → μ,log σ² → reparam → decoder)."""
    _torch()
    import torch
    import torch.nn as nn

    def _mlp(sizes):
        layers = []
        for i in range(len(sizes) - 1):
            layers.append(nn.Linear(sizes[i], sizes[i + 1]))
            if i < len(sizes) - 2:
                layers.append(nn.GELU())
        return nn.Sequential(*layers)

    class _VAE(nn.Module):
        def __init__(self):
            super().__init__()
            enc_sizes = [in_dim] + [hidden] * n_layers
            dec_sizes = [latent_dim] + [hidden] * n_layers + [in_dim]
            self.enc = _mlp(enc_sizes)
            self.fc_mu = nn.Linear(hidden, latent_dim)
            self.fc_logv = nn.Linear(hidden, latent_dim)
            self.dec = _mlp(dec_sizes)

        def encode(self, x):
            h = self.enc(x)
            return self.fc_mu(h), self.fc_logv(h)

        def reparam(self, mu, logv):
            std = torch.exp(0.5 * logv)
            eps = torch.randn_like(std)
            return mu + eps * std

        def decode(self, z):
            return self.dec(z)

        def forward(self, x):
            mu, logv = self.encode(x)
            z = self.reparam(mu, logv)
            return self.decode(z), mu, logv

    return _VAE()


def vae_loss(x_hat, x, mu, logv, *, beta: float = 1.0):
    """ELBO: reconstruction (MSE) + β·KL."""
    torch = _torch()
    rec = ((x_hat - x) ** 2).mean()
    kl = -0.5 * torch.mean(1 + logv - mu ** 2 - logv.exp())
    return rec + beta * kl, {"rec": float(rec.detach()), "kl": float(kl.detach())}


__all__ = ["VAE", "vae_loss"]
