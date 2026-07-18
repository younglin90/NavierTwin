"""Round 121 — Thomas algorithm + Crank-Nicolson heat."""

from __future__ import annotations

import numpy as np


class TestTridiagonal:
    def test_thomas_matches_solve(self) -> None:
        from naviertwin.core.linalg.tridiagonal import thomas_solve

        n = 20
        rng = np.random.default_rng(0)
        b = rng.uniform(2.0, 5.0, n)
        a = np.r_[0.0, rng.uniform(-1, 1, n - 1)]
        c = np.r_[rng.uniform(-1, 1, n - 1), 0.0]
        A = np.diag(a[1:], k=-1) + np.diag(b, k=0) + np.diag(c[:-1], k=1)
        d = rng.standard_normal(n)
        x = thomas_solve(a, b, c, d)
        assert np.allclose(A @ x, d, atol=1e-10)

    def test_crank_nicolson_stable_for_large_dt(self) -> None:
        """FTCS 는 발산하지만 CN 은 stable 한 dt 로 테스트."""
        from naviertwin.core.linalg.tridiagonal import crank_nicolson_heat_step

        nx = 51
        x = np.linspace(0, 1, nx)
        u = np.sin(np.pi * x)
        u[0] = u[-1] = 0.0
        dx = x[1] - x[0]
        nu = 0.1
        dt = 0.01  # r ≫ 0.5 → FTCS 불안정이지만 CN OK
        u_cur = u.copy()
        for _ in range(20):
            u_cur = crank_nicolson_heat_step(u_cur, nu, dx, dt)
        # 감쇠만 하고 발산하지 않음
        assert np.max(u_cur) < np.max(u)
        assert np.all(np.isfinite(u_cur))
