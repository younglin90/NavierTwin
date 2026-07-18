"""Round 236 — CV + grid search."""

from __future__ import annotations

import numpy as np


class TestCV:
    def test_kfold(self) -> None:
        from naviertwin.core.validation.cross_val import kfold_scores

        rng = np.random.default_rng(0)
        X = rng.standard_normal((50, 1))
        y = (X[:, 0] * 2 + 1) + 0.1 * rng.standard_normal(50)

        def fit_predict(X_tr, y_tr, X_val):
            m, b = np.polyfit(X_tr[:, 0], y_tr, 1)
            return m * X_val[:, 0] + b

        def mse(y_true, y_pred):
            return float(np.mean((y_true - y_pred) ** 2))

        scores = kfold_scores(X, y, fit_predict, mse, k=5)
        assert len(scores) == 5
        assert np.mean(scores) < 0.1

    def test_grid_search(self) -> None:
        from naviertwin.core.validation.cross_val import grid_search

        rng = np.random.default_rng(0)
        X = rng.standard_normal((60, 1))
        y = np.sin(3 * X[:, 0])

        def factory(params):
            deg = params["degree"]

            def fp(X_tr, y_tr, X_val):
                coefs = np.polyfit(X_tr[:, 0], y_tr, deg)
                return np.polyval(coefs, X_val[:, 0])
            return fp

        def mse(y_true, y_pred):
            return float(np.mean((y_true - y_pred) ** 2))

        res = grid_search(X, y, factory, {"degree": [1, 3, 5]}, mse, k=3)
        # 최적 degree 는 data-dependent; 단지 history 가 3개 이고 best 값 존재
        assert res["best"]["params"]["degree"] in (1, 3, 5)
        assert len(res["history"]) == 3
