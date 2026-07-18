"""Round 336 — Nelder-Mead."""

from __future__ import annotations

import numpy as np


class TestNelderMead:
    def test_quadratic(self) -> None:
        from naviertwin.core.optimization.nelder_mead import nelder_mead

        x = nelder_mead(lambda x: float(x @ x), x0=np.array([2.0, -3.0]))
        assert np.linalg.norm(x) < 1e-3

    def test_rosenbrock(self) -> None:
        from naviertwin.core.optimization.nelder_mead import nelder_mead

        def f(x):
            return float(100 * (x[1] - x[0] ** 2) ** 2 + (1 - x[0]) ** 2)

        x = nelder_mead(f, x0=np.array([-1.0, 1.0]), max_iter=2000)
        assert np.allclose(x, [1.0, 1.0], atol=1e-2)
