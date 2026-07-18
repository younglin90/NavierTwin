"""Round 342 — UKF."""

from __future__ import annotations

import numpy as np


class TestUKF:
    def test_constant_tracking(self) -> None:
        from naviertwin.core.data_assimilation.ukf import ukf_step

        rng = np.random.default_rng(0)
        x = np.zeros(1)
        P = np.array([[10.0]])
        true_x = 3.0
        for _ in range(50):
            z = np.array([true_x + rng.normal(0, 0.2)])
            x, P = ukf_step(
                x, P,
                f=lambda x: x, h=lambda x: x,
                z=z, Q=np.array([[0.01]]), R=np.array([[0.04]]),
            )
        assert abs(x[0] - true_x) < 0.3
        assert (np.linalg.eigvalsh(P) >= 0).all()

    def test_shapes(self) -> None:
        from naviertwin.core.data_assimilation.ukf import ukf_step

        x = np.zeros(2)
        P = np.eye(2)
        x2, P2 = ukf_step(
            x, P,
            f=lambda x: x ** 2,
            h=lambda x: x,
            z=np.array([1.0, 2.0]),
            Q=0.1 * np.eye(2), R=0.1 * np.eye(2),
        )
        assert x2.shape == (2,)
        assert P2.shape == (2, 2)
