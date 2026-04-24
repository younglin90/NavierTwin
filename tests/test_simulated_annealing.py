"""Round 225 — SA."""

from __future__ import annotations

import numpy as np


class TestSA:
    def test_quadratic(self) -> None:
        from naviertwin.core.optimization.simulated_annealing import sa

        x, f = sa(lambda v: float(v @ v), x0=np.array([5.0, -4.0]),
                  n_iter=3000, step=0.3, T0=1.0, cooling=0.995, seed=0)
        assert np.linalg.norm(x) < 0.5

    def test_multi_modal(self) -> None:
        """더 좋은 최소 탐색."""
        from naviertwin.core.optimization.simulated_annealing import sa

        def obj(v):
            return float((v[0] ** 2 - 1) ** 2 + v[1] ** 2)

        x, f = sa(obj, x0=np.array([0.5, 1.0]),
                  n_iter=3000, step=0.3, T0=1.0, seed=0)
        # 두 local min: (±1, 0)
        assert abs(abs(x[0]) - 1.0) < 0.3
        assert abs(x[1]) < 0.3
