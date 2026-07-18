"""Round 224 — PSO."""

from __future__ import annotations

import numpy as np


class TestPSO:
    def test_quadratic(self) -> None:
        from naviertwin.core.optimization.pso import pso

        x, f = pso(lambda v: float(v @ v),
                   bounds=[(-5, 5), (-5, 5)],
                   n_particles=30, n_iter=80, seed=0)
        assert np.linalg.norm(x) < 0.1

    def test_shifted(self) -> None:
        from naviertwin.core.optimization.pso import pso

        def obj(v):
            return float((v[0] - 3) ** 2 + (v[1] + 2) ** 2)

        x, f = pso(obj, bounds=[(-10, 10), (-10, 10)],
                   n_particles=30, n_iter=100, seed=0)
        assert abs(x[0] - 3) < 0.1
        assert abs(x[1] + 2) < 0.1
