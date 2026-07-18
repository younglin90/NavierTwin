"""Round 266 — ADI 2D heat."""

from __future__ import annotations

import numpy as np


class TestADI:
    def test_shape_preserved(self) -> None:
        from naviertwin.core.solvers.adi_2d import adi_step

        u = np.zeros((11, 11))
        u[5, 5] = 1.0
        u1 = adi_step(u, dt=0.01, dx=0.1, dy=0.1, alpha=1.0)
        assert u1.shape == u.shape
        # Dirichlet edge stays 0
        assert np.all(u1[0, :] == 0)
        assert np.all(u1[-1, :] == 0)
        assert np.all(u1[:, 0] == 0)
        assert np.all(u1[:, -1] == 0)

    def test_diffuses_peak(self) -> None:
        from naviertwin.core.solvers.adi_2d import adi_step

        u = np.zeros((21, 21))
        u[10, 10] = 1.0
        u_t = u.copy()
        for _ in range(20):
            u_t = adi_step(u_t, dt=0.005, dx=0.05, dy=0.05, alpha=1.0)
        # peak amplitude decreased, mass approx conserved (Dirichlet 0 → mass leaks slowly)
        assert u_t[10, 10] < 1.0
        assert u_t[10, 10] > 0.0
        # spread to neighbors
        assert u_t[9, 10] > 0
        assert u_t[10, 9] > 0

    def test_unconditional_stability(self) -> None:
        """ADI 는 무조건 안정 → 큰 dt 에도 발산 없음."""
        from naviertwin.core.solvers.adi_2d import adi_step

        rng = np.random.default_rng(0)
        u = np.zeros((15, 15))
        u[1:-1, 1:-1] = rng.standard_normal((13, 13))
        for _ in range(50):
            u = adi_step(u, dt=1.0, dx=0.1, dy=0.1, alpha=1.0)
        assert np.isfinite(u).all()
        assert np.max(np.abs(u)) < 100.0
