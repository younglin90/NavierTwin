"""Round 334 — NSGA-II constrained."""

from __future__ import annotations

import numpy as np


class TestNSGA2C:
    def test_feasible_solutions(self) -> None:
        from naviertwin.core.optimization.nsga2_constrained import nsga2_constrained

        rng = np.random.default_rng(0)
        # min (x², (x-2)²) s.t. x ≥ 1 (i.e. constraint c(x) = x - 1 ≥ 0)
        pop = nsga2_constrained(
            objectives=lambda x: np.array([x[0] ** 2, (x[0] - 2) ** 2]),
            constraints=lambda x: np.array([x[0] - 1.0]),
            n_pop=30, n_gen=30, dim=1,
            bounds=(np.array([-1.0]), np.array([3.0])), rng=rng,
        )
        assert pop.shape == (30, 1)
        # most population should satisfy x >= 1
        assert (pop[:, 0] >= 0.95).mean() > 0.7
