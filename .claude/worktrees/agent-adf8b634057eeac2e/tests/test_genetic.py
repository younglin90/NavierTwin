"""Round 228 — GA."""

from __future__ import annotations

import numpy as np


class TestGA:
    def test_quadratic(self) -> None:
        from naviertwin.core.optimization.genetic import ga

        x, f = ga(lambda v: float(v @ v),
                  bounds=[(-5, 5)] * 3,
                  n_pop=40, n_gen=80, seed=0)
        assert f < 0.1

    def test_shifted(self) -> None:
        from naviertwin.core.optimization.genetic import ga

        target = np.array([2.0, -1.0, 0.5])

        def obj(v):
            return float(np.sum((v - target) ** 2))

        x, f = ga(obj, bounds=[(-5, 5)] * 3,
                  n_pop=50, n_gen=100, seed=0)
        assert np.linalg.norm(x - target) < 0.3
