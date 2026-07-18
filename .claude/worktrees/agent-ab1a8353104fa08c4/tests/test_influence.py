"""Round 496 — influence function (linear regression)."""

from __future__ import annotations

import numpy as np


class TestInfluence:
    def test_shape(self) -> None:
        from naviertwin.utils.influence import linreg_influence

        rng = np.random.default_rng(0)
        X = rng.standard_normal((20, 3))
        y = X @ np.array([1.0, -1.0, 0.5]) + 0.1 * rng.standard_normal(20)
        infl = linreg_influence(X, y)
        assert infl.shape == (20, 3)

    def test_outlier_influential(self) -> None:
        from naviertwin.utils.influence import linreg_influence

        rng = np.random.default_rng(0)
        X = rng.standard_normal((30, 2))
        y = X @ np.array([1.0, 0.5])
        # one large outlier
        X_out = np.vstack([X, [[10.0, 10.0]]])
        y_out = np.concatenate([y, [100.0]])
        infl = linreg_influence(X_out, y_out)
        # last (outlier) has biggest |influence|
        norms = np.linalg.norm(infl, axis=1)
        # outlier influence should be in top quartile
        assert norms[-1] >= np.quantile(norms, 0.75)
