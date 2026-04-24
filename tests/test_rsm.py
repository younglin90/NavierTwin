"""Round 229 — RSM quadratic."""

from __future__ import annotations

import numpy as np


class TestRSM:
    def test_quadratic_exact(self) -> None:
        from naviertwin.core.surrogate.rsm import RSMQuadratic

        rng = np.random.default_rng(0)
        X = rng.standard_normal((80, 2))
        y = 1.0 + 2 * X[:, 0] - X[:, 1] + 3 * X[:, 0] ** 2 + X[:, 0] * X[:, 1]
        rsm = RSMQuadratic().fit(X, y)
        assert rsm.r_squared(X, y) > 0.9999

    def test_predict(self) -> None:
        from naviertwin.core.surrogate.rsm import RSMQuadratic

        rng = np.random.default_rng(0)
        X = rng.standard_normal((30, 3))
        y = np.sum(X ** 2, axis=1)
        rsm = RSMQuadratic().fit(X, y)
        y_hat = rsm.predict(X)
        assert np.allclose(y_hat, y, atol=1e-8)
