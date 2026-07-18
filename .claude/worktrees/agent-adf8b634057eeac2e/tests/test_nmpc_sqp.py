"""Round 348 — Nonlinear MPC."""

from __future__ import annotations

import numpy as np


class TestNMPC:
    def test_first_action_drives_state(self) -> None:
        from naviertwin.core.control.nmpc_sqp import nmpc_solve

        # x_{k+1} = x + u, want x → 0.  cost = x² + 0.01 u²
        def f(x, u):
            return x + u

        def cost(x, u):
            return float(x[0] ** 2 + 0.01 * u[0] ** 2)

        u = nmpc_solve(f, cost, x0=np.array([1.0]), N=10, n_iter=300, lr=0.01)
        # first action negative (push x down)
        assert u[0, 0] < -0.1

    def test_shape(self) -> None:
        from naviertwin.core.control.nmpc_sqp import nmpc_solve

        u = nmpc_solve(
            lambda x, u: x + u,
            lambda x, u: float(x[0] ** 2),
            x0=np.array([1.0]), N=5, n_u=1, n_iter=10,
        )
        assert u.shape == (5, 1)
