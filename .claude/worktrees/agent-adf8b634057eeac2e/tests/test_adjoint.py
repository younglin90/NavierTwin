"""Round 131 — adjoint sensitivity."""

from __future__ import annotations

import numpy as np


class TestAdjoint:
    def test_linear_vs_fd(self) -> None:
        """A(p) = diag(p) + 1, b = [1,1,...], J = sum(u²) / 2."""
        from naviertwin.core.optimization.adjoint import (
            fd_sensitivity,
            linear_adjoint_sensitivity,
        )

        n = 4

        def A_fn(p):
            return np.diag(p) + np.ones((n, n)) * 0.1

        def b_fn(p):
            return np.ones(n) + 0.0 * p  # p-independent

        def J_fn(u, p):  # noqa: ARG001
            return 0.5 * float(u @ u)

        def forward(p):
            A = A_fn(p)
            b = b_fn(p)
            u = np.linalg.solve(A, b)
            return J_fn(u, p)

        p0 = np.array([2.0, 3.0, 4.0, 5.0])
        J_a, g_a = linear_adjoint_sensitivity(A_fn, b_fn, J_fn, p0)
        J_f, g_f = fd_sensitivity(forward, p0, eps=1e-5)
        assert abs(J_a - J_f) < 1e-8
        assert np.allclose(g_a, g_f, rtol=1e-3, atol=1e-5)
