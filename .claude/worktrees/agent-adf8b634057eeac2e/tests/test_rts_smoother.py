"""Round 199 — RTS smoother."""

from __future__ import annotations

import numpy as np


class TestRTS:
    def test_smoother_better_than_filter(self) -> None:
        from naviertwin.core.data_assimilation.kalman import KalmanFilter
        from naviertwin.core.data_assimilation.rts_smoother import rts_smoother

        rng = np.random.default_rng(0)
        F = np.array([[1.0, 0.1], [0.0, 1.0]])
        H = np.array([[1.0, 0.0]])
        Q = np.eye(2) * 1e-4
        R = np.array([[0.05]])

        # true trajectory + noisy obs
        T = 100
        x = np.array([0.0, 1.0])
        xs_true = np.zeros((T + 1, 2))
        xs_true[0] = x
        zs = np.zeros((T, 1))
        for k in range(T):
            x = F @ x + rng.multivariate_normal([0, 0], Q)
            xs_true[k + 1] = x
            zs[k] = H @ x + rng.normal(0, np.sqrt(R[0, 0]))

        # Filter
        kf = KalmanFilter(F=F, H=H, Q=Q, R=R, x0=np.zeros(2), P0=np.eye(2))
        xf = np.zeros((T + 1, 2))
        for k in range(T):
            kf.predict()
            kf.update(zs[k])
            xf[k + 1] = kf.x

        # Smoother
        xs, _ = rts_smoother(F, H, Q, R, np.zeros(2), np.eye(2), zs)

        # 스무더 RMSE ≤ 필터 RMSE
        rmse_filter = np.sqrt(np.mean((xf - xs_true) ** 2))
        rmse_smoother = np.sqrt(np.mean((xs - xs_true) ** 2))
        assert rmse_smoother <= rmse_filter + 1e-6

    def test_shapes(self) -> None:
        from naviertwin.core.data_assimilation.rts_smoother import rts_smoother

        F = np.eye(2)
        H = np.eye(2)
        Q = np.eye(2) * 0.01
        R = np.eye(2) * 0.1
        z = np.zeros((5, 2))
        xs, Ps = rts_smoother(F, H, Q, R, np.zeros(2), np.eye(2), z)
        assert xs.shape == (6, 2)
        assert Ps.shape == (6, 2, 2)
