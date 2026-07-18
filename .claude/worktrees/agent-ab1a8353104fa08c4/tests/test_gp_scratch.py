"""Round 163 — scratch GP."""

from __future__ import annotations

import numpy as np


class TestGP:
    def test_interpolates_training(self) -> None:
        from naviertwin.core.surrogate.gp_scratch import GPRegressor

        X = np.linspace(0, 1, 15)[:, None]
        y = np.sin(X[:, 0] * 4)
        gp = GPRegressor(lengthscale=0.15, sigma=1.0, noise=1e-8).fit(X, y)
        mu, _ = gp.predict(X)
        assert np.allclose(mu, y, atol=1e-4)

    def test_test_point_variance_nonneg(self) -> None:
        from naviertwin.core.surrogate.gp_scratch import GPRegressor

        X = np.linspace(0, 1, 10)[:, None]
        y = np.sin(X[:, 0] * 4)
        gp = GPRegressor(lengthscale=0.2, sigma=1.0).fit(X, y)
        mu, var = gp.predict(np.array([[0.33], [5.0]]))
        assert np.all(var >= 0)
        # 먼 점 분산 > 훈련 분산
        assert var[1] > var[0]

    def test_lml_finite(self) -> None:
        from naviertwin.core.surrogate.gp_scratch import GPRegressor

        X = np.random.default_rng(0).standard_normal((20, 2))
        y = X[:, 0] + X[:, 1]
        gp = GPRegressor(lengthscale=1.0, sigma=1.0).fit(X, y)
        lml = gp.log_marginal_likelihood()
        assert np.isfinite(lml)
