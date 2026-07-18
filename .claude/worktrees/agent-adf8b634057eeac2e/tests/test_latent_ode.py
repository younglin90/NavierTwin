"""Round 401 — latent ODE."""

from __future__ import annotations

import pytest


class TestLatentODE:
    def test_encode_decode(self) -> None:
        torch = pytest.importorskip("torch")
        from naviertwin.core.dimensionality_reduction.nonlinear.latent_ode import (
            LatentODE,
        )

        m = LatentODE(in_dim=10, latent_dim=3, hidden=16)
        x = torch.randn(5, 10)
        z = m.encode(x)
        x_rec = m.decode(z)
        assert z.shape == (5, 3)
        assert x_rec.shape == x.shape

    def test_step_advances(self) -> None:
        torch = pytest.importorskip("torch")
        from naviertwin.core.dimensionality_reduction.nonlinear.latent_ode import (
            LatentODE,
        )

        m = LatentODE(in_dim=4, latent_dim=2, hidden=8)
        z = torch.randn(2, 2)
        z2 = m.step(z, dt=0.1)
        assert z2.shape == z.shape
