"""Round 331 — trust region."""

from __future__ import annotations

import numpy as np


class TestTrustRegion:
    def test_quadratic_1d(self) -> None:
        from naviertwin.core.optimization.trust_region import trust_region_minimize

        x = trust_region_minimize(
            lambda x: float((x[0] - 3.0) ** 2),
            lambda x: np.array([2 * (x[0] - 3.0)]),
            x0=np.array([0.0]),
            max_iter=50,
        )
        assert abs(x[0] - 3.0) < 1e-3

    def test_quadratic_2d(self) -> None:
        from naviertwin.core.optimization.trust_region import trust_region_minimize

        A = np.diag([2.0, 1.0])
        b = np.array([1.0, -1.0])
        x = trust_region_minimize(
            lambda x: 0.5 * x @ A @ x - b @ x,
            lambda x: A @ x - b,
            x0=np.zeros(2),
            max_iter=100,
        )
        # solution: A x = b → [0.5, -1.0]
        assert np.allclose(x, [0.5, -1.0], atol=1e-3)
