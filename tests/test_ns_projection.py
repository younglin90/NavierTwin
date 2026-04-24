"""Round 143 — 2D NS projection cavity."""

from __future__ import annotations

import numpy as np


class TestNS:
    def test_cavity_runs(self) -> None:
        from naviertwin.core.solvers.ns_projection_2d import solve_cavity

        u, v, p = solve_cavity(nx=16, ny=16, Re=100.0, n_steps=30)
        assert u.shape == (16, 16)
        # 상단 lid 속도가 적용되어 내부에 유동 생성
        assert np.max(np.abs(u)) > 0
        assert np.all(np.isfinite(u))
        assert np.all(np.isfinite(v))
        assert np.all(np.isfinite(p))

    def test_bc_walls_noslip(self) -> None:
        from naviertwin.core.solvers.ns_projection_2d import solve_cavity

        u, v, _ = solve_cavity(nx=12, ny=12, Re=50.0, n_steps=10)
        # 좌/우/하단 벽 = 0 (모서리는 lid 에 의해 1이 될 수 있으므로 내부만)
        assert np.max(np.abs(u[0, :-1])) < 1e-12
        assert np.max(np.abs(u[-1, :-1])) < 1e-12
        assert np.max(np.abs(u[:, 0])) < 1e-12
        # 상단 lid
        assert np.allclose(u[:, -1], 1.0)
