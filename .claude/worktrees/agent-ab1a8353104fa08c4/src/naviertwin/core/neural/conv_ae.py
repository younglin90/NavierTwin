"""2D Convolutional Autoencoder 블록 — CFD 필드 이미지 ROM용.

Examples:
    >>> import torch  # doctest: +SKIP
    >>> from naviertwin.core.neural.conv_ae import ConvAutoencoder  # doctest: +SKIP
"""

from __future__ import annotations


def _torch():
    try:
        import torch

        return torch
    except ImportError as exc:
        raise RuntimeError("torch 필요") from exc


def ConvAutoencoder(
    in_channels: int = 1, latent_dim: int = 32,
    base: int = 16, input_size: int = 32,
):
    """대칭 인코더/디코더 (downsample by 4)."""
    _torch()
    import torch.nn as nn

    if input_size % 4 != 0:
        raise ValueError("input_size must be divisible by 4")

    mid = input_size // 4

    class _AE(nn.Module):
        def __init__(self):
            super().__init__()
            self.enc = nn.Sequential(
                nn.Conv2d(in_channels, base, 3, stride=2, padding=1), nn.GELU(),
                nn.Conv2d(base, 2 * base, 3, stride=2, padding=1), nn.GELU(),
            )
            self.flatten = nn.Flatten()
            self.to_latent = nn.Linear(2 * base * mid * mid, latent_dim)
            self.from_latent = nn.Linear(latent_dim, 2 * base * mid * mid)
            self.unflatten_shape = (2 * base, mid, mid)
            self.dec = nn.Sequential(
                nn.ConvTranspose2d(2 * base, base, 4, stride=2, padding=1), nn.GELU(),
                nn.ConvTranspose2d(base, in_channels, 4, stride=2, padding=1),
            )

        def encode(self, x):
            h = self.enc(x)
            return self.to_latent(self.flatten(h))

        def decode(self, z):
            h = self.from_latent(z).view(-1, *self.unflatten_shape)
            return self.dec(h)

        def forward(self, x):
            return self.decode(self.encode(x))

    return _AE()


__all__ = ["ConvAutoencoder"]
