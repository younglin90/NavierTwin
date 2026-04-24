"""Round 171 — SINDy."""

from __future__ import annotations

import numpy as np


class TestSINDy:
    def test_identifies_linear_decay(self) -> None:
        from naviertwin.core.system_id.sindy import polynomial_library, stls

        t = np.linspace(0, 10, 1000)
        x = np.exp(-0.5 * t)
        dx = -0.5 * x
        Theta = polynomial_library(x[:, None], degree=2)
        Xi = stls(Theta, dx[:, None], threshold=0.05)
        # x 항만 살아남고 계수 ≈ -0.5
        # columns: [1, x0, x0*x0]
        assert abs(Xi[1, 0] - (-0.5)) < 0.05
        assert abs(Xi[0, 0]) < 0.05  # bias 작음
        assert abs(Xi[2, 0]) < 0.05  # x² 작음

    def test_library_sizes(self) -> None:
        from naviertwin.core.system_id.sindy import polynomial_library

        X = np.random.default_rng(0).standard_normal((30, 2))
        Theta = polynomial_library(X, degree=3, include_bias=True)
        # 1 + 2 + 3 + 4 = 10 (1, x, y, x², xy, y², x³, x²y, xy², y³)
        assert Theta.shape == (30, 10)

    def test_lorenz_sparse(self) -> None:
        """Lorenz-like: ẋ = σ(y-x) → ẋ ≈ -σ x + σ y."""
        from naviertwin.core.system_id.sindy import polynomial_library, stls

        rng = np.random.default_rng(0)
        n = 1000
        x = rng.uniform(-2, 2, n)
        y = rng.uniform(-2, 2, n)
        sigma = 10.0
        dx = sigma * (y - x)
        X = np.stack([x, y], axis=1)
        Theta = polynomial_library(X, degree=2)
        Xi = stls(Theta, dx[:, None], threshold=0.2)
        # columns: [1, x, y, x², xy, y²]
        assert abs(Xi[1, 0] - (-sigma)) < 1.0
        assert abs(Xi[2, 0] - sigma) < 1.0
