"""Autoencoder-ROM with latent-ODE — encode → ODE in z → decode.

Examples:
    >>> import torch
    >>> from naviertwin.core.dimensionality_reduction.nonlinear.latent_ode import (
    ...     LatentODE,
    ... )
    >>> m = LatentODE(in_dim=8, latent_dim=2, hidden=16)
    >>> x = torch.randn(4, 8)
    >>> z = m.encode(x)
    >>> z.shape
    torch.Size([4, 2])
"""

from __future__ import annotations

import torch
from torch import nn


class LatentODE(nn.Module):
    def __init__(self, in_dim: int, latent_dim: int = 4, hidden: int = 32) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.SiLU(),
            nn.Linear(hidden, latent_dim),
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden), nn.SiLU(),
            nn.Linear(hidden, in_dim),
        )
        # ODE field f: z -> dz/dt
        self.f = nn.Sequential(
            nn.Linear(latent_dim, hidden), nn.SiLU(),
            nn.Linear(hidden, latent_dim),
        )

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)

    def step(self, z: torch.Tensor, dt: float = 0.1) -> torch.Tensor:
        """RK4 step in latent."""
        k1 = self.f(z)
        k2 = self.f(z + 0.5 * dt * k1)
        k3 = self.f(z + 0.5 * dt * k2)
        k4 = self.f(z + dt * k3)
        return z + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)


__all__ = ["LatentODE"]
