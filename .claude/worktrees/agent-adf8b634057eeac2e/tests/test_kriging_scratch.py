"""Round 232 — Ordinary Kriging."""

from __future__ import annotations

import numpy as np


class TestKrig:
    def test_interp_training(self) -> None:
        from naviertwin.core.surrogate.kriging_scratch import OrdinaryKriging

        X = np.linspace(0, 1, 15)[:, None]
        y = np.sin(5 * X[:, 0])
        k = OrdinaryKriging(theta=0.15, nugget=1e-10).fit(X, y)
        yh = k.predict(X)
        assert np.allclose(yh, y, atol=1e-3)

    def test_predict_var_nonneg(self) -> None:
        from naviertwin.core.surrogate.kriging_scratch import OrdinaryKriging

        X = np.linspace(0, 1, 10)[:, None]
        y = np.cos(3 * X[:, 0])
        k = OrdinaryKriging(theta=0.2, nugget=1e-6).fit(X, y)
        var = k.predict_var(np.array([[0.5], [2.0]]))
        assert np.all(var >= 0)
