"""Round 367 — HLLC Riemann."""

from __future__ import annotations

import numpy as np


class TestHLLC:
    def test_uniform_zero_flux_difference(self) -> None:
        from naviertwin.core.solvers.hllc_1d import hllc_flux

        # uniform L=R → flux = analytical flux
        U = np.array([1.0, 0.5, 2.0])
        F = hllc_flux(U, U, gamma=1.4)
        # ρu = 0.5, ρu²+p = 0.25 + (γ-1)(E - ρu²/2)
        assert F[0] == 0.5

    def test_sod_shape(self) -> None:
        from naviertwin.core.solvers.hllc_1d import hllc_flux

        UL = np.array([1.0, 0.0, 2.5])
        UR = np.array([0.125, 0.0, 0.25])
        F = hllc_flux(UL, UR, gamma=1.4)
        assert F.shape == (3,)
        assert np.isfinite(F).all()
