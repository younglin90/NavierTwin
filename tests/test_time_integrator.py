"""Round 111 — ODE 시간 적분."""

from __future__ import annotations

import numpy as np
import pytest


class TestIntegrator:
    @pytest.mark.parametrize("method", ["euler", "rk2", "rk4"])
    def test_exp_decay(self, method: str) -> None:
        from naviertwin.core.analysis.time_integrator import integrate_ode

        def f(t, y):  # noqa: ARG001
            return -y

        t, y = integrate_ode(
            f, np.array([1.0]), (0.0, 1.0), dt=0.001, method=method,
        )
        tol = 1e-2 if method == "euler" else 1e-5
        assert abs(y[-1, 0] - np.exp(-1.0)) < tol

    def test_harmonic_oscillator_rk4(self) -> None:
        from naviertwin.core.analysis.time_integrator import integrate_ode

        def f(t, y):  # noqa: ARG001
            return np.array([y[1], -y[0]])

        t, y = integrate_ode(
            f, np.array([1.0, 0.0]), (0, 2 * np.pi), dt=1e-3, method="rk4",
        )
        # 한 주기 후 초기 상태 근처 (endpoint overshoot ~ O(dt))
        assert abs(y[-1, 0] - 1.0) < 1e-2
        assert abs(y[-1, 1]) < 1e-2

    def test_invalid(self) -> None:
        from naviertwin.core.analysis.time_integrator import integrate_ode

        with pytest.raises(ValueError):
            integrate_ode(lambda t, y: y, np.zeros(1), (0, 1), 0.1, method="bogus")

    def test_scipy_adaptive(self) -> None:
        pytest.importorskip("scipy")
        from naviertwin.core.analysis.time_integrator import integrate_scipy

        def f(t, y):  # noqa: ARG001
            return -y

        ts, ys = integrate_scipy(
            f, np.array([1.0]), (0, 1), method="RK45",
        )
        assert abs(ys[-1, 0] - np.exp(-1)) < 1e-6
