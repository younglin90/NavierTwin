"""Round 267 — IMEX RK2."""

from __future__ import annotations

import numpy as np


class TestIMEX:
    def test_decay_linear(self) -> None:
        """du/dt = -10 u (stiff), exact: u(t)=u0 e^{-10t}."""
        from naviertwin.core.solvers.imex_rk import imex_rk2_step

        L = -10.0 * np.eye(2)
        u = np.array([1.0, 1.0])
        dt = 0.05
        for _ in range(20):
            u = imex_rk2_step(u, lambda x: np.zeros_like(x), L, dt=dt)
        # exact at t=1.0: e^{-10}
        assert np.allclose(u, np.exp(-10.0), atol=0.01)

    def test_explicit_part(self) -> None:
        """L=0, f_E(u)=-u → exponential decay via explicit only."""
        from naviertwin.core.solvers.imex_rk import imex_rk2_step

        L = np.zeros((1, 1))
        u = np.array([1.0])
        for _ in range(100):
            u = imex_rk2_step(u, lambda x: -x, L, dt=0.01)
        assert np.allclose(u, np.exp(-1.0), atol=0.01)

    def test_finite(self) -> None:
        from naviertwin.core.solvers.imex_rk import imex_rk2_step

        L = -100.0 * np.eye(3)
        u = np.array([1.0, -1.0, 0.5])
        for _ in range(50):
            u = imex_rk2_step(u, lambda x: 0.1 * np.sin(x), L, dt=0.1)
        assert np.isfinite(u).all()
