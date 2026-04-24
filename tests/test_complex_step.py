"""Round 103 — Complex-step 미분."""

from __future__ import annotations

import numpy as np


class TestComplexStep:
    def test_scalar(self) -> None:
        from naviertwin.utils.complex_step import cs_derivative

        d = cs_derivative(lambda x: x ** 3, 2.0)
        assert abs(d - 12.0) < 1e-10

    def test_gradient(self) -> None:
        from naviertwin.utils.complex_step import cs_gradient

        def f(x):
            return np.sum(x ** 3)

        g = cs_gradient(f, np.array([1.0, 2.0, 3.0]))
        assert np.allclose(g, [3.0, 12.0, 27.0], atol=1e-10)

    def test_jacobian(self) -> None:
        from naviertwin.utils.complex_step import cs_jacobian

        A = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])

        def f(x):
            return A @ x

        J = cs_jacobian(f, np.array([1.0, 1.0]))
        assert J.shape == (3, 2)
        assert np.allclose(J, A, atol=1e-10)

    def test_exp_sin(self) -> None:
        from naviertwin.utils.complex_step import cs_gradient

        def f(x):
            return np.exp(np.sin(x[0])) + x[1] * x[1]

        x0 = np.array([0.5, 1.5])
        g = cs_gradient(f, x0)
        # d/dx0 = exp(sin x0) * cos x0
        expected0 = np.exp(np.sin(0.5)) * np.cos(0.5)
        expected1 = 2 * 1.5
        assert abs(g[0] - expected0) < 1e-10
        assert abs(g[1] - expected1) < 1e-10
