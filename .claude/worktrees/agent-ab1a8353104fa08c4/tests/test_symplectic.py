"""Round 175 — symplectic integrators."""

from __future__ import annotations

import numpy as np


class TestSymp:
    def test_energy_conservation_verlet(self) -> None:
        """조화진동자 H = ½(p² + q²) 보존."""
        from naviertwin.core.analysis.symplectic import velocity_verlet

        q, p = np.array([1.0]), np.array([0.0])
        qs, ps = velocity_verlet(lambda q: -q, q, p, dt=0.01, n=5000)
        H = 0.5 * (qs[:, 0] ** 2 + ps[:, 0] ** 2)
        assert abs(H.max() - H.min()) < 1e-3

    def test_leapfrog(self) -> None:
        from naviertwin.core.analysis.symplectic import leapfrog

        qs, ps = leapfrog(lambda q: -q, np.array([1.0]), np.array([0.0]),
                          dt=0.01, n=1000)
        assert qs.shape == (1001, 1)
        H = 0.5 * (qs[:, 0] ** 2 + ps[:, 0] ** 2)
        assert abs(H.max() - H.min()) < 1e-3

    def test_2d_system(self) -> None:
        """2D 조화: F = -q (각 성분)."""
        from naviertwin.core.analysis.symplectic import velocity_verlet

        q0 = np.array([1.0, 0.5])
        p0 = np.array([0.0, 0.3])
        qs, ps = velocity_verlet(lambda q: -q, q0, p0, dt=0.01, n=2000)
        H = 0.5 * (np.sum(qs ** 2, axis=1) + np.sum(ps ** 2, axis=1))
        assert abs(H.max() - H.min()) < 1e-3
