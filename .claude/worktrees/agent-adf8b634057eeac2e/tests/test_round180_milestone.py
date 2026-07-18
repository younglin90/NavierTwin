"""Round 180 — 마일스톤 R171-R179."""

from __future__ import annotations

import numpy as np
import pytest

R171_179 = [
    "naviertwin.core.system_id.sindy",
    "naviertwin.core.system_id.dmdc",
    "naviertwin.core.tools.mesh_refine_1d",
    "naviertwin.core.neural.gnn_layer",
    "naviertwin.core.analysis.symplectic",
    "naviertwin.core.analysis.implicit_ode",
    "naviertwin.core.system_id.reservoir",
    "naviertwin.core.surrogate.bayesian_linear",
    "naviertwin.core.monitoring.anomaly",
]


class TestRound180:
    @pytest.mark.parametrize("m", R171_179)
    def test_importable(self, m: str) -> None:
        import importlib
        importlib.import_module(m)

    def test_dmdc_to_blr_pipeline(self) -> None:
        """DMDc 로 선형모델 식별 → 잔차를 BLR 로 회귀 → 이상 탐지."""
        from naviertwin.core.monitoring.anomaly import threshold_detector
        from naviertwin.core.surrogate.bayesian_linear import (
            BayesianLinearRegression,
        )
        from naviertwin.core.system_id.dmdc import fit_dmdc, rollout_dmdc

        rng = np.random.default_rng(0)
        A = np.array([[0.95, 0.05], [-0.1, 0.9]])
        B = np.array([[1.0], [0.5]])
        T = 150
        X = np.zeros((2, T))
        U = rng.standard_normal((1, T - 1)) * 0.5
        for k in range(T - 1):
            X[:, k + 1] = A @ X[:, k] + B[:, 0] * U[0, k]
        # 일부 구간에 anomaly 주입
        X[:, 100:110] += 10.0

        # fit on clean part
        A_hat, B_hat = fit_dmdc(X[:, :90], U[:, :89])
        pred = rollout_dmdc(A_hat, B_hat, X[:, 0], U)
        residuals = np.linalg.norm(pred[:, 1:] - X[:, 1:], axis=0)
        alarms = threshold_detector(residuals, threshold=1.0)
        assert alarms[100:].any()

        # BLR smooth: residuals vs time
        t = np.arange(residuals.size)
        Phi = np.stack([np.ones_like(t), t], axis=1).astype(float)
        blr = BayesianLinearRegression(alpha=1e-2, beta=1.0).fit(Phi, residuals)
        assert blr.mean_ is not None

    def test_sindy_dmdc_coexist(self) -> None:
        from naviertwin.core.system_id.dmdc import fit_dmdc
        from naviertwin.core.system_id.sindy import polynomial_library, stls

        rng = np.random.default_rng(0)
        x = rng.uniform(-1, 1, 200)
        dx = -0.5 * x
        Theta = polynomial_library(x[:, None], degree=2)
        Xi = stls(Theta, dx[:, None], threshold=0.05)
        assert abs(Xi[1, 0] - (-0.5)) < 0.1

        X = np.vstack([x, dx])
        U = np.zeros((1, X.shape[1] - 1))
        A, B = fit_dmdc(X, U)
        assert A.shape == (2, 2)
