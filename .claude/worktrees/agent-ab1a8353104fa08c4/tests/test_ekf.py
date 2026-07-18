"""Round 341 — EKF."""

from __future__ import annotations

import numpy as np


class TestEKF:
    def test_linear_recovers_kalman(self) -> None:
        from naviertwin.core.data_assimilation.ekf import ekf_step

        x = np.zeros(2)
        P = np.eye(2)
        F = np.array([[1.0, 1.0], [0.0, 1.0]])
        H = np.array([[1.0, 0.0]])
        Q = 0.01 * np.eye(2)
        R = 0.1 * np.eye(1)
        for k in range(10):
            x, P = ekf_step(
                x, P,
                f=lambda x: F @ x,
                F=lambda x: F,
                h=lambda x: H @ x,
                H=lambda x: H,
                z=np.array([float(k + 1)]),
                Q=Q, R=R,
            )
        assert np.isfinite(x).all()
        assert (np.linalg.eigvalsh(P) >= 0).all()

    def test_position_tracking(self) -> None:
        from naviertwin.core.data_assimilation.ekf import ekf_step

        rng = np.random.default_rng(0)
        x = np.zeros(1)
        P = np.eye(1) * 10.0
        true_x = 5.0
        for _ in range(50):
            z = np.array([true_x + rng.normal(0, 0.1)])
            x, P = ekf_step(
                x, P,
                f=lambda x: x, F=lambda x: np.eye(1),
                h=lambda x: x, H=lambda x: np.eye(1),
                z=z, Q=0.01 * np.eye(1), R=0.01 * np.eye(1),
            )
        assert abs(x[0] - true_x) < 0.2
