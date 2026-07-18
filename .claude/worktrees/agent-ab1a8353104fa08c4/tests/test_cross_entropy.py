"""Round 227 — CEM."""

from __future__ import annotations

import numpy as np


class TestCEM:
    def test_quadratic_neg(self) -> None:
        """CEM 은 maximize → -x·x 를 최대화 = x=0 찾기."""
        from naviertwin.core.uncertainty.cross_entropy import cem_optimize

        x, f = cem_optimize(lambda v: -float(v @ v),
                            dim=3, n_samples=100, elite_frac=0.2,
                            n_iter=40, seed=0)
        assert np.linalg.norm(x) < 0.1

    def test_shifted(self) -> None:
        from naviertwin.core.uncertainty.cross_entropy import cem_optimize

        target = np.array([2.0, -1.5])

        def obj(v):
            return -float(np.sum((v - target) ** 2))

        x, f = cem_optimize(obj, dim=2, n_samples=100, n_iter=50, seed=0)
        assert np.linalg.norm(x - target) < 0.2
