"""Round 113 — linear Kalman filter."""

from __future__ import annotations

import numpy as np


class TestKalman:
    def test_denoise_constant(self) -> None:
        from naviertwin.core.data_assimilation.kalman import (
            KalmanFilter,
            run_filter,
        )

        rng = np.random.default_rng(0)
        true_val = 2.5
        z = true_val + rng.normal(0, 0.2, size=(200, 1))
        kf = KalmanFilter(
            F=np.eye(1), H=np.eye(1),
            Q=np.eye(1) * 1e-6, R=np.eye(1) * 0.04,
            x0=np.array([0.0]), P0=np.eye(1),
        )
        hist = run_filter(kf, z)
        # 안정된 값에 가깝게 수렴
        assert abs(hist[-1, 0] - true_val) < 0.1

    def test_linear_tracking(self) -> None:
        from naviertwin.core.data_assimilation.kalman import KalmanFilter

        # 1D 위치/속도 추적: x = [p, v], p_{k+1} = p_k + v_k dt
        dt = 0.1
        F = np.array([[1.0, dt], [0.0, 1.0]])
        H = np.array([[1.0, 0.0]])
        kf = KalmanFilter(
            F=F, H=H,
            Q=np.eye(2) * 1e-4, R=np.array([[0.01]]),
            x0=np.array([0.0, 0.0]), P0=np.eye(2),
        )
        # 속도 1로 움직이는 물체 관측
        for k in range(100):
            true_pos = k * dt * 1.0
            kf.predict()
            kf.update(np.array([true_pos]))
        assert abs(kf.x[1] - 1.0) < 0.1  # 속도 추정
