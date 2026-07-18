"""Round 186 — 역설계."""

from __future__ import annotations

import numpy as np


class TestInvDesign:
    def test_scalar_recovery(self) -> None:
        from naviertwin.core.optimization.inverse_design import inverse_design

        def fwd(p):
            return np.array([p[0] ** 2])

        p, hist = inverse_design(fwd, np.array([4.0]), np.array([0.5]),
                                 lr=0.1, n_iter=500)
        assert abs(p[0] - 2.0) < 0.05
        assert hist[-1] < hist[0]

    def test_vector_with_reg(self) -> None:
        from naviertwin.core.optimization.inverse_design import inverse_design

        A = np.array([[1.0, 0.5], [0.2, 1.0]])

        def fwd(p):
            return A @ p

        target = np.array([1.0, 1.0])
        p, _ = inverse_design(fwd, target, np.zeros(2), reg=1e-3, lr=0.2, n_iter=500)
        # target 근접
        assert np.linalg.norm(A @ p - target) < 0.05

    def test_bounds(self) -> None:
        from naviertwin.core.optimization.inverse_design import inverse_design

        def fwd(p):
            return p ** 2

        p, _ = inverse_design(
            fwd, np.array([100.0]), np.array([5.0]),
            bounds=(None, np.array([3.0])), lr=0.1, n_iter=200,
        )
        assert p[0] <= 3.0 + 1e-6
