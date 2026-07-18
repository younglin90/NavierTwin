"""Round 200 — MEGA 마일스톤: R191-R199 import + Kalman-smoother + PID-plant e2e."""

from __future__ import annotations

import numpy as np
import pytest

R191_199 = [
    "naviertwin.core.neural.rnn_blocks",
    "naviertwin.core.neural.pinn_trainer",
    "naviertwin.core.analysis.reynolds",
    "naviertwin.core.solvers.cfl",
    "naviertwin.core.analysis.dbscan",
    "naviertwin.core.analysis.kmeans",
    "naviertwin.core.control.pid",
    "naviertwin.core.control.mpc_linear",
    "naviertwin.core.data_assimilation.rts_smoother",
]


class TestRound200:
    @pytest.mark.parametrize("m", R191_199)
    def test_importable(self, m: str) -> None:
        import importlib
        importlib.import_module(m)

    def test_pid_control_plant(self) -> None:
        """PID 로 1차 plant 를 set-point tracking."""
        from naviertwin.core.control.pid import PID
        from naviertwin.core.solvers.cfl import cfl_number

        p = PID(kp=1.2, ki=0.5, kd=0.05)
        y = 0.0
        dt = 0.1
        hist = []
        for _ in range(150):
            u = p.step(setpoint=1.0, measurement=y, dt=dt)
            y = 0.9 * y + 0.1 * u
            hist.append(y)
        assert abs(hist[-1] - 1.0) < 0.05
        # CFL 유틸 확인
        assert cfl_number(dt, 0.1, 1.0) == pytest.approx(1.0)

    def test_kmeans_dbscan_coexist(self) -> None:
        from naviertwin.core.analysis.dbscan import dbscan, n_clusters
        from naviertwin.core.analysis.kmeans import kmeans

        rng = np.random.default_rng(0)
        c1 = rng.standard_normal((40, 2)) * 0.1
        c2 = rng.standard_normal((40, 2)) * 0.1 + np.array([5.0, 5.0])
        X = np.vstack([c1, c2])
        centers, labels = kmeans(X, k=2, seed=0)
        db_labels = dbscan(X, eps=0.3, min_samples=3)
        assert centers.shape == (2, 2)
        assert n_clusters(db_labels) >= 2

    def test_rts_filter_smoother(self) -> None:
        from naviertwin.core.data_assimilation.rts_smoother import rts_smoother

        F = np.eye(1)
        H = np.eye(1)
        Q = np.array([[1e-4]])
        R = np.array([[1e-2]])
        rng = np.random.default_rng(0)
        z = np.sin(np.linspace(0, np.pi, 20))[:, None] + rng.normal(0, 0.1, (20, 1))
        xs, _ = rts_smoother(F, H, Q, R, np.zeros(1), np.eye(1), z)
        assert xs.shape == (21, 1)
