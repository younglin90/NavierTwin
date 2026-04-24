"""Round 332 — SQP."""

from __future__ import annotations

import numpy as np


class TestSQP:
    def test_quadratic_eq(self) -> None:
        from naviertwin.core.optimization.sqp import sqp_eq

        x = sqp_eq(
            f=lambda x: float(x @ x),
            grad=lambda x: 2 * x,
            hess=lambda x: 2 * np.eye(2),
            h=lambda x: np.array([x[0] + x[1] - 1.0]),
            hjac=lambda x: np.array([[1.0, 1.0]]),
            x0=np.zeros(2),
        )
        assert np.allclose(x, [0.5, 0.5], atol=1e-6)

    def test_two_constraints(self) -> None:
        """min ‖x-c‖² s.t. x_0=1, x_1=2 → (1,2,c_2)."""
        from naviertwin.core.optimization.sqp import sqp_eq

        c = np.array([5.0, 5.0, 5.0])
        x = sqp_eq(
            f=lambda x: float((x - c) @ (x - c)),
            grad=lambda x: 2 * (x - c),
            hess=lambda x: 2 * np.eye(3),
            h=lambda x: np.array([x[0] - 1.0, x[1] - 2.0]),
            hjac=lambda x: np.array([[1.0, 0, 0], [0, 1.0, 0]]),
            x0=np.zeros(3),
        )
        assert np.allclose(x, [1.0, 2.0, 5.0], atol=1e-6)
