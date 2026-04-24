"""Round 129 — Newton / Broyden."""

from __future__ import annotations

import numpy as np


class TestNonlinear:
    def test_newton_scalar(self) -> None:
        from naviertwin.core.linalg.nonlinear import newton_solve

        def F(x):
            return np.array([x[0] ** 2 - 2.0])

        def J(x):
            return np.array([[2 * x[0]]])

        x, info = newton_solve(F, J, x0=np.array([1.0]))
        assert info["converged"]
        assert abs(x[0] - np.sqrt(2)) < 1e-8

    def test_newton_system(self) -> None:
        from naviertwin.core.linalg.nonlinear import newton_solve

        def F(x):
            return np.array([
                x[0] + x[1] - 3.0,
                x[0] * x[1] - 2.0,
            ])

        def J(x):
            return np.array([[1, 1], [x[1], x[0]]], dtype=float)

        x, info = newton_solve(F, J, x0=np.array([0.5, 2.5]))
        assert info["converged"]
        assert np.linalg.norm(F(x)) < 1e-8

    def test_broyden(self) -> None:
        from naviertwin.core.linalg.nonlinear import broyden_solve

        def F(x):
            return np.array([
                x[0] ** 2 + x[1] ** 2 - 1.0,
                x[0] - x[1],
            ])

        x, info = broyden_solve(F, x0=np.array([0.8, 0.6]))
        assert info["converged"]
        assert abs(x[0] - x[1]) < 1e-6

    def test_fd_jacobian(self) -> None:
        from naviertwin.core.linalg.nonlinear import fd_jacobian

        def F(x):
            return np.array([x[0] ** 2, x[1] * x[0]])

        J = fd_jacobian(F, np.array([2.0, 3.0]), eps=1e-5)
        expected = np.array([[4.0, 0.0], [3.0, 2.0]])
        assert np.allclose(J, expected, atol=1e-4)
