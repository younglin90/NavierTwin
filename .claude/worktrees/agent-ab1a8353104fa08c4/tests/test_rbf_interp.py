"""Round 135 — RBF 보간."""

from __future__ import annotations

import numpy as np
import pytest


class TestRBF:
    @pytest.mark.parametrize("kernel", ["gaussian", "multiquadric", "inverse_multiquadric"])
    def test_interpolates_centers(self, kernel: str) -> None:
        from naviertwin.core.analysis.rbf_interp import RBFInterpolator

        rng = np.random.default_rng(0)
        X = rng.standard_normal((20, 2))
        y = rng.standard_normal(20)
        rbf = RBFInterpolator(X, y, kernel=kernel, epsilon=0.5, reg=1e-10)
        y_hat = rbf(X)
        # regularization 으로 정확 보간은 아니고 근사
        assert np.max(np.abs(y_hat - y)) < 0.01

    def test_smooth_function(self) -> None:
        from naviertwin.core.analysis.rbf_interp import RBFInterpolator

        rng = np.random.default_rng(0)
        X = rng.uniform(-1, 1, size=(30, 2))
        y = X[:, 0] ** 2 + X[:, 1] ** 2
        rbf = RBFInterpolator(X, y, kernel="multiquadric", epsilon=1.0)
        X_test = rng.uniform(-0.5, 0.5, size=(10, 2))
        y_true = X_test[:, 0] ** 2 + X_test[:, 1] ** 2
        y_pred = rbf(X_test)
        assert np.max(np.abs(y_pred - y_true)) < 0.1
