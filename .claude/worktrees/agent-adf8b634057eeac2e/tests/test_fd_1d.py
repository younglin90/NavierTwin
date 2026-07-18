"""Round 119 — 1D FD 솔버."""

from __future__ import annotations

import numpy as np


class TestFD1D:
    def test_heat_decay(self) -> None:
        """sin 초기조건의 열방정식은 e^{-ν π² t} 로 감쇠."""
        from naviertwin.core.solvers.fd_1d import solve_heat_1d

        nu = 0.01
        T = 0.5
        x, t, U = solve_heat_1d(nx=64, L=1.0, T=T, nu=nu)
        # 진폭 = max(u)
        a0 = np.max(np.abs(U[:, 0]))
        a1 = np.max(np.abs(U[:, -1]))
        expected = a0 * np.exp(-nu * (np.pi ** 2) * T)
        assert abs(a1 - expected) / expected < 0.05

    def test_burgers_evolves(self) -> None:
        from naviertwin.core.solvers.fd_1d import solve_burgers_1d

        x, t, U = solve_burgers_1d(nx=128, T=0.1, nu=0.01)
        # 경계는 0 유지
        assert np.max(np.abs(U[0, :])) < 1e-12
        assert np.max(np.abs(U[-1, :])) < 1e-12
        # sin → 충격파 형성으로 프로파일이 비대칭이 됨
        # 에너지는 감소
        e0 = float(np.sum(U[:, 0] ** 2))
        eT = float(np.sum(U[:, -1] ** 2))
        assert eT < e0
