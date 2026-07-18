"""Round 130 — LM / Gauss-Newton."""

from __future__ import annotations

import numpy as np


class TestLeastSquares:
    def test_lm_exponential(self) -> None:
        from naviertwin.core.linalg.least_squares import levenberg_marquardt

        t = np.linspace(0, 1, 30)
        rng = np.random.default_rng(0)
        y = 2.0 * np.exp(-3.0 * t) + 0.02 * rng.standard_normal(30)

        def r(p):
            return p[0] * np.exp(-p[1] * t) - y

        p, info = levenberg_marquardt(r, p0=np.array([1.0, 1.0]), max_iter=200)
        assert info["converged"] or info["cost"] < 0.01
        assert abs(p[0] - 2.0) < 0.1
        assert abs(p[1] - 3.0) < 0.3

    def test_gauss_newton_linear_regression(self) -> None:
        from naviertwin.core.linalg.least_squares import gauss_newton

        # y = a x + b
        rng = np.random.default_rng(0)
        x = np.linspace(0, 1, 50)
        y = 3.0 * x + 1.5 + 0.01 * rng.standard_normal(50)

        def r(p):
            return p[0] * x + p[1] - y

        p, _ = gauss_newton(r, p0=np.array([0.0, 0.0]))
        assert abs(p[0] - 3.0) < 0.05
        assert abs(p[1] - 1.5) < 0.05
