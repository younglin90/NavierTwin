"""Round 176 — implicit ODE integrators."""

from __future__ import annotations

import numpy as np


class TestImplicit:
    def test_implicit_euler_stiff(self) -> None:
        """stiff linear system: y' = -1000 y, explicit Euler divergence 조건."""
        from naviertwin.core.analysis.implicit_ode import implicit_euler_linear

        A = np.array([[-1000.0]])
        ts, ys = implicit_euler_linear(A, np.array([1.0]), (0, 0.01), dt=1e-3)
        # dt=1e-3 는 explicit 불안정이지만 implicit 은 OK
        assert np.all(np.isfinite(ys))
        assert ys[-1, 0] < 0.5

    def test_crank_nicolson_accuracy(self) -> None:
        """y' = -y, y(0)=1 → y(1)=e⁻¹."""
        from naviertwin.core.analysis.implicit_ode import crank_nicolson_linear

        A = np.array([[-1.0]])
        ts, ys = crank_nicolson_linear(A, np.array([1.0]), (0, 1), dt=0.01)
        assert abs(ys[-1, 0] - np.exp(-1)) < 1e-4

    def test_implicit_nonlinear(self) -> None:
        """y' = -y³ → y(t) = y0 / sqrt(1 + 2 y0² t)."""
        from naviertwin.core.analysis.implicit_ode import implicit_euler_nonlinear

        def f(t, y):  # noqa: ARG001
            return -(y ** 3)

        ts, ys = implicit_euler_nonlinear(f, np.array([1.0]), (0, 1), dt=0.01)
        y_exact = 1.0 / np.sqrt(1 + 2 * 1)
        assert abs(ys[-1, 0] - y_exact) < 0.02
