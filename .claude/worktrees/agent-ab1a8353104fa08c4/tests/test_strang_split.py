"""Round 396 — Strang splitting."""

from __future__ import annotations

import numpy as np


class TestStrang:
    def test_combined_decay(self) -> None:
        from naviertwin.core.solvers.strang_split import strang_step

        # A: u' = -u (exact: e^{-dt} u);  B: u' = -2 u (exact: e^{-2dt} u)
        # combined: u' = -3 u
        u = np.array([1.0])
        for _ in range(20):
            u = strang_step(
                u,
                op_A=lambda u, dt: u * np.exp(-dt),
                op_B=lambda u, dt: u * np.exp(-2 * dt),
                dt=0.05,
            )
        # exact at t=1.0: e^{-3} ≈ 0.0498
        assert abs(u[0] - np.exp(-3.0)) < 1e-3

    def test_shape(self) -> None:
        from naviertwin.core.solvers.strang_split import strang_step

        u = np.zeros(5)
        u2 = strang_step(u, lambda u, dt: u, lambda u, dt: u + dt, dt=0.1)
        assert u2.shape == u.shape
