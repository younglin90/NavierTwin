"""Round 255 — FV 1D conservation."""

from __future__ import annotations

import numpy as np


class TestFV1D:
    def test_mass_conservation_linear(self) -> None:
        """∂u/∂t + ∂u/∂x = 0 periodic → mass 보존."""
        from naviertwin.core.solvers.fv_1d import solve_conservation_1d

        x, _, U = solve_conservation_1d(n_cells=80, T=0.5, flux=lambda u: u)
        m0 = float(U[:, 0].sum())
        mT = float(U[:, -1].sum())
        assert abs(mT - m0) < 1e-8

    def test_burgers_like(self) -> None:
        from naviertwin.core.solvers.fv_1d import solve_conservation_1d

        x, _, U = solve_conservation_1d(
            n_cells=100, T=0.1, flux=lambda u: 0.5 * u ** 2, c_max=1.0,
        )
        assert np.all(np.isfinite(U))
        # periodic sin → 전체 질량 보존
        assert abs(U[:, 0].sum() - U[:, -1].sum()) < 1e-6

    def test_upwind_scheme(self) -> None:
        from naviertwin.core.solvers.fv_1d import solve_conservation_1d

        x, _, U = solve_conservation_1d(
            n_cells=60, T=0.05, scheme="upwind", flux=lambda u: u,
        )
        assert np.all(np.isfinite(U))
