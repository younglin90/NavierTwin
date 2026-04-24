"""Round 350 — I category milestone: control/DA + RLS + EKF-LQR e2e."""

from __future__ import annotations

import numpy as np


class TestMilestoneI:
    def test_imports(self) -> None:
        from naviertwin.core.control import (  # noqa: F401
            adaptive_pid,
            lqg,
            lqr,
            nmpc_sqp,
        )
        from naviertwin.core.data_assimilation import (  # noqa: F401
            ekf,
            envar,
            iterated_ekf,
            mhe,
            rls,
            ukf,
        )

    def test_rls_recovers_linear(self) -> None:
        from naviertwin.core.data_assimilation.rls import RLS

        rng = np.random.default_rng(0)
        true_theta = np.array([2.0, -1.0])
        rls = RLS(n_features=2, lam=1.0, delta=100.0)
        for _ in range(200):
            x = rng.standard_normal(2)
            y = float(true_theta @ x) + rng.normal(0, 0.01)
            rls.update(x, y)
        assert np.allclose(rls.theta, true_theta, atol=0.05)

    def test_ekf_lqr_pipeline(self) -> None:
        """EKF estimate → LQR feedback stabilizes a linear system with noisy meas."""
        from naviertwin.core.control.lqr import lqr_gain
        from naviertwin.core.data_assimilation.ekf import ekf_step

        A = np.array([[1.05, 0.05], [0.0, 1.0]])
        B = np.array([[0.0], [0.05]])
        C = np.array([[1.0, 0.0]])
        K = lqr_gain(A, B, np.eye(2), np.eye(1))
        rng = np.random.default_rng(0)
        x = np.array([3.0, 0.0])
        x_hat = np.zeros(2)
        P = np.eye(2)
        for _ in range(100):
            u = -K @ x_hat
            # plant
            x = A @ x + B @ u + 0.001 * rng.standard_normal(2)
            y = C @ x + 0.05 * rng.standard_normal(1)
            # estimator
            x_hat, P = ekf_step(
                x_hat, P,
                f=lambda x: A @ x + B @ u, F=lambda x: A,
                h=lambda x: C @ x, H=lambda x: C,
                z=y, Q=0.01 * np.eye(2), R=0.01 * np.eye(1),
            )
        assert np.linalg.norm(x) < 1.0
