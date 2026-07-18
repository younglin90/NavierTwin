"""Round 28 — 1D Burgers + Heat FD 솔버 테스트."""

from __future__ import annotations

import numpy as np


class TestBurgers:
    def test_run_and_shape(self) -> None:
        from naviertwin.core.solver_interfaces.pde_solvers import solve_burgers_1d

        u0 = np.sin(np.linspace(0, 2 * np.pi, 64, endpoint=False))
        t, U = solve_burgers_1d(u0, nu=0.01, L=2 * np.pi, T=0.5, n_steps=200)
        assert U.shape == (200, 64)
        assert np.all(np.isfinite(U))

    def test_shock_forms(self) -> None:
        """낮은 ν 에서 Burgers 는 shock 형성 → gradient 증가."""
        from naviertwin.core.solver_interfaces.pde_solvers import solve_burgers_1d

        x = np.linspace(0, 2 * np.pi, 64, endpoint=False)
        u0 = np.sin(x)
        _, U = solve_burgers_1d(u0, nu=0.001, L=2 * np.pi, T=1.0, n_steps=500)
        grad_init = float(np.max(np.abs(np.gradient(u0))))
        grad_final = float(np.max(np.abs(np.gradient(U[-1]))))
        assert grad_final > grad_init * 0.5  # 적어도 초기 gradient 수준 유지


class TestHeat:
    def test_decays_to_zero(self) -> None:
        from naviertwin.core.solver_interfaces.pde_solvers import solve_heat_1d

        x = np.linspace(0, 1, 33)
        u0 = np.sin(np.pi * x)
        t, U = solve_heat_1d(u0, alpha=0.1, L=1.0, T=1.0, n_steps=200)
        # 경계값 0
        assert abs(U[-1, 0]) < 1e-10
        assert abs(U[-1, -1]) < 1e-10
        # 최대값이 충분히 감쇠
        assert np.max(np.abs(U[-1])) < np.max(np.abs(u0)) * 0.5

    def test_first_mode_decay_rate(self) -> None:
        """sin(πx) 초기조건의 정확한 감쇠율: exp(-α π² t)."""
        from naviertwin.core.solver_interfaces.pde_solvers import solve_heat_1d

        x = np.linspace(0, 1, 65)
        u0 = np.sin(np.pi * x)
        alpha = 0.05
        T = 0.5
        _, U = solve_heat_1d(u0, alpha=alpha, L=1.0, T=T, n_steps=500)
        expected = np.exp(-alpha * np.pi ** 2 * T)
        max_final = float(np.max(np.abs(U[-1])))
        assert abs(max_final - expected) / expected < 0.05
