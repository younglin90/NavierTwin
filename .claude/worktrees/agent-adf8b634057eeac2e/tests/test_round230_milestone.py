"""Round 230 — 마일스톤 R221-R229."""

from __future__ import annotations

import numpy as np
import pytest

R221_229 = [
    "naviertwin.core.uncertainty.conformal",
    "naviertwin.core.neural.positional_enc",
    "naviertwin.core.surrogate.rff_regression",
    "naviertwin.core.optimization.pso",
    "naviertwin.core.optimization.simulated_annealing",
    "naviertwin.core.optimization.cma_es_simple",
    "naviertwin.core.uncertainty.cross_entropy",
    "naviertwin.core.optimization.genetic",
    "naviertwin.core.surrogate.rsm",
]


class TestRound230:
    @pytest.mark.parametrize("m", R221_229)
    def test_importable(self, m: str) -> None:
        import importlib
        importlib.import_module(m)

    def test_rff_then_conformal(self) -> None:
        """RFF 회귀 후 conformal 구간 → coverage."""
        from naviertwin.core.surrogate.rff_regression import RFFRegression
        from naviertwin.core.uncertainty.conformal import SplitConformal

        rng = np.random.default_rng(0)
        X = rng.uniform(-1, 1, (300, 1))
        y = np.sin(3 * X[:, 0]) + 0.1 * rng.standard_normal(300)

        X_tr, X_cal, X_test = X[:150], X[150:250], X[250:]
        y_tr, y_cal, y_test = y[:150], y[150:250], y[250:]

        r = RFFRegression(num_features=128, sigma=0.3, ridge=1e-3, seed=0).fit(X_tr, y_tr)
        y_cal_hat = r.predict(X_cal)
        cp = SplitConformal(alpha=0.1).calibrate(y_cal, y_cal_hat)
        y_test_hat = r.predict(X_test)
        cov = cp.coverage(y_test, y_test_hat)
        assert 0.7 <= cov <= 1.0

    def test_optimizer_comparison(self) -> None:
        from naviertwin.core.optimization.genetic import ga
        from naviertwin.core.optimization.pso import pso

        def obj(v):
            return float(v @ v)

        x_pso, f_pso = pso(obj, bounds=[(-3, 3)] * 2, n_particles=25, n_iter=50, seed=0)
        x_ga, f_ga = ga(obj, bounds=[(-3, 3)] * 2, n_pop=25, n_gen=50, seed=0)
        assert f_pso < 0.5 and f_ga < 0.5
