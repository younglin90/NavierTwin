"""Round 397 — SSP-RK3."""

from __future__ import annotations

import numpy as np


class TestSSPRK3:
    def test_decay(self) -> None:
        from naviertwin.core.solvers.ssp_rk3 import ssp_rk3_step

        u = np.array([1.0])
        for _ in range(100):
            u = ssp_rk3_step(u, lambda u: -u, dt=0.01)
        # exact: e^{-1.0}
        assert abs(u[0] - np.exp(-1.0)) < 1e-4

    def test_third_order_accuracy(self) -> None:
        from naviertwin.core.solvers.ssp_rk3 import ssp_rk3_step

        # solve u' = -u from 1.0 to 0.1 with two step sizes
        u_exact = np.exp(-0.1)
        errs = []
        for n in [10, 20]:
            dt = 0.1 / n
            u = np.array([1.0])
            for _ in range(n):
                u = ssp_rk3_step(u, lambda u: -u, dt=dt)
            errs.append(abs(u[0] - u_exact))
        # ratio ≈ 8 for 3rd-order (halving dt → 1/8 error)
        ratio = errs[0] / max(errs[1], 1e-30)
        assert ratio > 4
