"""Round 368 — AUSM+."""

from __future__ import annotations

import numpy as np


class TestAUSM:
    def test_finite_sod(self) -> None:
        from naviertwin.core.solvers.ausm_plus import ausm_plus_flux

        UL = np.array([1.0, 0.0, 2.5])
        UR = np.array([0.125, 0.0, 0.25])
        F = ausm_plus_flux(UL, UR, gamma=1.4)
        assert F.shape == (3,)
        assert np.isfinite(F).all()

    def test_uniform_no_motion(self) -> None:
        from naviertwin.core.solvers.ausm_plus import ausm_plus_flux

        # uniform stationary: ρ=1, u=0, p ⇒ F = (0, p, 0)
        rho, p = 1.0, 1.0
        gamma = 1.4
        E = p / (gamma - 1)
        U = np.array([rho, 0.0, E])
        F = ausm_plus_flux(U, U, gamma=gamma)
        # mass flux 0, energy flux 0
        assert abs(F[0]) < 1e-12
        assert abs(F[2]) < 1e-12
        # momentum = p
        assert abs(F[1] - p) < 1e-10
