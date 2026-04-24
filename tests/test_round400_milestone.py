"""Round 400 — N category milestone: high-order numerics e2e."""

from __future__ import annotations

import numpy as np


class TestMilestoneN:
    def test_imports(self) -> None:
        from naviertwin.core.solvers import (  # noqa: F401
            ausm_plus,
            entropy_stable,
            hllc_1d,
            low_mach,
            muscl,
            ppm,
            roe,
            ssp_rk3,
            strang_split,
            tvb_limiter,
            weno5,
        )

    def test_ssp_rk3_advection_e2e(self) -> None:
        """1D linear advection u_t + u_x = 0 with SSP-RK3 + upwind."""
        from naviertwin.core.solvers.ssp_rk3 import ssp_rk3_step

        n = 41
        x = np.linspace(0, 1, n)
        u = np.exp(-50 * (x - 0.3) ** 2)
        dx = x[1] - x[0]
        dt = 0.5 * dx
        c = 1.0

        def rhs(u):
            r = np.zeros_like(u)
            r[1:] = -c * (u[1:] - u[:-1]) / dx
            return r

        for _ in range(int(0.4 / dt)):
            u = ssp_rk3_step(u, rhs, dt=dt)
        # peak should have moved to ~0.7
        peak_x = x[np.argmax(u)]
        assert 0.6 < peak_x < 0.8

    def test_weno5_value(self) -> None:
        from naviertwin.core.solvers.weno5 import weno5_recon_left

        # smooth: linear → exact
        u = np.array([0., 1., 2., 3., 4.])
        v = weno5_recon_left(u)
        assert np.isclose(v, 2.5)
