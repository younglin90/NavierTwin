"""Round 178 — Bayesian linear regression."""

from __future__ import annotations

import numpy as np


class TestBLR:
    def test_recovers_true(self) -> None:
        from naviertwin.core.surrogate.bayesian_linear import BayesianLinearRegression

        rng = np.random.default_rng(0)
        X = np.linspace(0, 1, 100)[:, None]
        Phi = np.hstack([np.ones_like(X), X])
        w_true = np.array([1.5, 3.0])
        y = Phi @ w_true + 0.05 * rng.standard_normal(100)
        blr = BayesianLinearRegression(alpha=1e-2, beta=100.0).fit(Phi, y)
        assert blr.mean_ is not None
        assert np.allclose(blr.mean_, w_true, atol=0.05)

    def test_predictive_variance(self) -> None:
        from naviertwin.core.surrogate.bayesian_linear import BayesianLinearRegression

        rng = np.random.default_rng(0)
        X = rng.uniform(-1, 1, size=(20, 1))
        Phi = np.hstack([np.ones_like(X), X])
        y = 2 * X[:, 0] + rng.standard_normal(20) * 0.1
        blr = BayesianLinearRegression(alpha=1.0, beta=10.0).fit(Phi, y)
        _, var = blr.predict(Phi)
        assert np.all(var > 0)

    def test_samples(self) -> None:
        from naviertwin.core.surrogate.bayesian_linear import BayesianLinearRegression

        Phi = np.eye(4)
        y = np.array([1.0, 2.0, 3.0, 4.0])
        blr = BayesianLinearRegression(alpha=0.1, beta=10.0).fit(Phi, y)
        samples = blr.sample_weights(5)
        assert samples.shape == (5, 4)
