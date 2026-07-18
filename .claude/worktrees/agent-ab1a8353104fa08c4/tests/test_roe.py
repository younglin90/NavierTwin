"""Round 394 — Roe flux."""

from __future__ import annotations

import numpy as np


class TestRoe:
    def test_uniform(self) -> None:
        from naviertwin.core.solvers.roe import roe_flux

        U = np.array([1.0, 0.5, 2.0])
        F = roe_flux(U, U, gamma=1.4)
        # uniform → physical flux (no dissipation since UL=UR)
        rho_u = U[1]  # 0.5
        assert np.isclose(F[0], rho_u)

    def test_finite_sod(self) -> None:
        from naviertwin.core.solvers.roe import roe_flux

        UL = np.array([1.0, 0.0, 2.5])
        UR = np.array([0.125, 0.0, 0.25])
        F = roe_flux(UL, UR, gamma=1.4)
        assert F.shape == (3,)
        assert np.isfinite(F).all()
