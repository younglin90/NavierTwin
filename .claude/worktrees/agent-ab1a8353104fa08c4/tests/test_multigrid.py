"""Round 261 — multigrid V-cycle."""

from __future__ import annotations

import numpy as np


class TestMG:
    def test_sin_sin(self) -> None:
        """-∇²u = 2π² sin πx sin πy → u = sin πx sin πy, Dirichlet 0."""
        from naviertwin.core.linalg.multigrid import solve_poisson_multigrid

        n = 33
        L = 1.0
        h = L / (n - 1)
        x = np.linspace(0, L, n)
        X, Y = np.meshgrid(x, x, indexing="ij")
        u_true = np.sin(np.pi * X) * np.sin(np.pi * Y)
        f = 2 * (np.pi ** 2) * u_true  # -Δu = f
        u, info = solve_poisson_multigrid(f, h, max_cycles=30, tol=1e-6, levels=4)
        err = np.max(np.abs(u - u_true))
        assert err < 0.05  # 33x33 에서 이산화 오차 수준

    def test_residual_decreases(self) -> None:
        from naviertwin.core.linalg.multigrid import solve_poisson_multigrid

        rng = np.random.default_rng(0)
        n = 33
        f = rng.standard_normal((n, n))
        f[0, :] = f[-1, :] = f[:, 0] = f[:, -1] = 0
        u, info = solve_poisson_multigrid(f, h=1 / (n - 1), max_cycles=20, levels=4)
        # random rhs → 잔차가 초기값보다 감소
        r_init = float(np.linalg.norm(f))
        assert info["residual"] < r_init
