"""Round 120 — 2D 열방정식 솔버."""

from __future__ import annotations

import numpy as np


class TestFD2D:
    def test_decay(self) -> None:
        """sin(πx)sin(πy) 초기조건 → exp(-2νπ²t) 감쇠."""
        from naviertwin.core.solvers.fd_2d import solve_heat_2d

        nu = 0.05
        T = 0.1
        x, y, t, U = solve_heat_2d(nx=32, ny=32, T=T, nu=nu)
        a0 = np.max(np.abs(U[:, :, 0]))
        aT = np.max(np.abs(U[:, :, -1]))
        expected = a0 * np.exp(-2 * nu * np.pi ** 2 * T)
        assert abs(aT - expected) / expected < 0.1

    def test_boundary_zero(self) -> None:
        from naviertwin.core.solvers.fd_2d import solve_heat_2d

        _, _, _, U = solve_heat_2d(nx=16, ny=16, T=0.01, nu=0.01)
        assert np.max(np.abs(U[0, :, :])) < 1e-12
        assert np.max(np.abs(U[-1, :, :])) < 1e-12
        assert np.max(np.abs(U[:, 0, :])) < 1e-12
        assert np.max(np.abs(U[:, -1, :])) < 1e-12
