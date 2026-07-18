"""Round 333 — augmented Lagrangian."""

from __future__ import annotations

import numpy as np


class TestAugLagrangian:
    def test_quadratic_eq(self) -> None:
        from naviertwin.core.optimization.aug_lagrangian import aug_lagrangian

        x = aug_lagrangian(
            f=lambda x: float(x @ x),
            grad=lambda x: 2 * x,
            h=lambda x: np.array([x[0] + x[1] - 1.0]),
            hjac=lambda x: np.array([[1.0, 1.0]]),
            x0=np.zeros(2),
            mu=10.0, n_outer=15,
        )
        assert np.allclose(x, [0.5, 0.5], atol=1e-3)
