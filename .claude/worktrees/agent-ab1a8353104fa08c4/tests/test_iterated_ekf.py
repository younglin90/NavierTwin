"""Round 343 — Iterated EKF."""

from __future__ import annotations

import numpy as np


class TestIEKF:
    def test_nonlinear_measurement(self) -> None:
        from naviertwin.core.data_assimilation.iterated_ekf import iekf_step

        # state x; measurement z = x²
        x = np.array([0.5])
        P = np.eye(1) * 1.0
        for _ in range(20):
            x, P = iekf_step(
                x, P,
                f=lambda x: x, F=lambda x: np.eye(1),
                h=lambda x: x ** 2, H=lambda x: 2 * x[:, None].T,
                z=np.array([4.0]),  # true x = 2
                Q=0.01 * np.eye(1), R=0.01 * np.eye(1),
                n_iter=5,
            )
        assert abs(x[0] - 2.0) < 0.3
        assert (np.linalg.eigvalsh(P) >= 0).all()
