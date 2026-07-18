"""Round 134 — BFGS."""

from __future__ import annotations

import numpy as np


class TestBFGS:
    def test_rosenbrock(self) -> None:
        from naviertwin.core.optimization.bfgs import bfgs_minimize

        def f(x):
            return float((1 - x[0]) ** 2 + 100 * (x[1] - x[0] ** 2) ** 2)

        def g(x):
            return np.array([
                -2 * (1 - x[0]) - 400 * x[0] * (x[1] - x[0] ** 2),
                200 * (x[1] - x[0] ** 2),
            ])

        x, info = bfgs_minimize(f, g, np.array([-1.2, 1.0]), max_iter=1000, tol=1e-6)
        assert info["converged"]
        assert np.linalg.norm(x - np.array([1.0, 1.0])) < 1e-3

    def test_quadratic(self) -> None:
        from naviertwin.core.optimization.bfgs import bfgs_minimize

        Q = np.diag([1.0, 10.0, 100.0])

        def f(x):
            return 0.5 * float(x @ Q @ x)

        def g(x):
            return Q @ x

        x, info = bfgs_minimize(f, g, np.array([5.0, 5.0, 5.0]))
        assert info["converged"]
        assert np.linalg.norm(x) < 1e-4
