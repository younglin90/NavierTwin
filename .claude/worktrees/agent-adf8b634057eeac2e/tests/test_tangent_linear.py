"""Round 256 — tangent-linear."""

from __future__ import annotations

import numpy as np


class TestTL:
    def test_directional_quadratic(self) -> None:
        from naviertwin.core.optimization.tangent_linear import directional_derivative

        def f(x):
            # complex-safe: return complex scalar (not cast to float)
            return x[0] ** 2 + x[1] ** 2

        d = directional_derivative(f, np.array([1.0, 2.0]), np.array([1.0, 0.0]))
        assert abs(d - 2.0) < 1e-10

    def test_jvp_fd(self) -> None:
        from naviertwin.core.optimization.tangent_linear import jvp_fd

        A = np.array([[1.0, 2.0], [3.0, 4.0]])
        v = np.array([1.0, 0.0])
        out = jvp_fd(lambda x: A @ x, np.array([1.0, 1.0]), v)
        assert np.allclose(out, A[:, 0], atol=1e-5)

    def test_gradient_from_jvp(self) -> None:
        from naviertwin.core.optimization.tangent_linear import gradient_from_jvp

        def f(x):
            return np.sum(x ** 3)  # complex-safe

        g = gradient_from_jvp(f, np.array([1.0, 2.0, 3.0]))
        assert np.allclose(g, [3, 12, 27], atol=1e-10)
