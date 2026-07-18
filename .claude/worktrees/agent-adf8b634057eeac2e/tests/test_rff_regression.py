"""Round 223 — RFF regression."""

from __future__ import annotations

import numpy as np


class TestRFF:
    def test_sine_fit(self) -> None:
        from naviertwin.core.surrogate.rff_regression import RFFRegression

        rng = np.random.default_rng(0)
        X = rng.uniform(-1, 1, (200, 1))
        y = np.sin(4 * X[:, 0])
        r = RFFRegression(num_features=256, sigma=0.25, ridge=1e-4, seed=0).fit(X, y)
        y_hat = r.predict(X)
        corr = float(np.corrcoef(y_hat, y)[0, 1])
        assert corr > 0.95

    def test_generalize(self) -> None:
        from naviertwin.core.surrogate.rff_regression import RFFRegression

        rng = np.random.default_rng(0)
        X = rng.uniform(-1, 1, (100, 2))
        y = X[:, 0] ** 2 + X[:, 1] ** 2
        r = RFFRegression(num_features=256, sigma=0.5, ridge=1e-3).fit(X, y)
        X_t = rng.uniform(-0.5, 0.5, (20, 2))
        y_t = X_t[:, 0] ** 2 + X_t[:, 1] ** 2
        y_pred = r.predict(X_t)
        assert np.mean((y_pred - y_t) ** 2) < 0.1
