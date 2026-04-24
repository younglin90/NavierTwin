"""Round 154 — 4D-Var cost function."""

from __future__ import annotations

import numpy as np


class TestVar4D:
    def test_linear_advection(self) -> None:
        """simple scalar: M(x) = α x, H = identity."""
        from naviertwin.core.data_assimilation.var4d_cost import (
            fd_gradient,
            var4d_cost,
        )

        alpha = 0.95
        xb = np.array([1.0])
        B_inv = np.array([[1.0]])
        R_inv = np.array([[10.0]])

        def M(x, t):  # noqa: ARG001
            return alpha * x

        def H(x):
            return x

        # true x0 = 2, observe over 5 steps + noise
        rng = np.random.default_rng(0)
        x_true = np.array([2.0])
        obs = []
        x = x_true.copy()
        for t in range(5):
            x = alpha * x
            obs.append(H(x) + rng.normal(0, 0.05, size=1))

        def cost(x0):
            return var4d_cost(x0, xb, B_inv, M, H, obs, R_inv)

        J_true = cost(x_true)
        J_far = cost(np.array([5.0]))
        assert J_true < J_far

        g = fd_gradient(x_true, cost)
        assert abs(g[0]) < 10 * abs(fd_gradient(np.array([5.0]), cost)[0])

    def test_optimum_at_true(self) -> None:
        """gradient at true state < gradient at wrong state."""
        from naviertwin.core.data_assimilation.var4d_cost import (
            var4d_cost,
        )

        xb = np.array([0.0])
        B_inv = np.array([[0.01]])  # weak prior
        R_inv = np.array([[100.0]])  # strong obs

        def M(x, t):  # noqa: ARG001
            return x  # identity model

        def H(x):
            return x

        obs = [np.array([3.0])] * 10

        def cost(x0):
            return var4d_cost(x0, xb, B_inv, M, H, obs, R_inv)

        # 관측이 매우 강하면 optimum ≈ 3
        J_at_3 = cost(np.array([3.0]))
        J_at_5 = cost(np.array([5.0]))
        assert J_at_3 < J_at_5
