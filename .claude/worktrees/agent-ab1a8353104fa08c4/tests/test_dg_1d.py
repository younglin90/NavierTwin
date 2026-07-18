"""Round 219 — 1D DG advection."""

from __future__ import annotations

import numpy as np


class TestDG:
    def test_runs_stable(self) -> None:
        from naviertwin.core.solvers.dg_1d import solve_advection_1d_dg

        x, t, U = solve_advection_1d_dg(n_cells=16, T=0.1, c=1.0, p=2)
        assert U.shape[1] > 0
        assert np.all(np.isfinite(U))

    def test_short_time_bounded(self) -> None:
        """짧은 시간 동안 값이 발산하지 않음."""
        from naviertwin.core.solvers.dg_1d import solve_advection_1d_dg

        x, _, U = solve_advection_1d_dg(
            n_cells=32, T=0.05, c=1.0, p=2,
            u0=lambda xx: np.sin(2 * np.pi * xx),
        )
        assert np.all(np.isfinite(U))
