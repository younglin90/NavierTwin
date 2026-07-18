"""Round 403 — SINDYc."""

from __future__ import annotations

import numpy as np


class TestSINDYc:
    def test_recover_linear(self) -> None:
        from naviertwin.core.system_id.sindyc import sindyc_fit

        rng = np.random.default_rng(0)
        X = rng.uniform(-1, 1, 200).reshape(-1, 1)
        U = rng.uniform(-1, 1, 200).reshape(-1, 1)
        # ẋ = -2 x + 0.5 u
        Xdot = -2.0 * X + 0.5 * U
        Xi = sindyc_fit(X, Xdot, U, threshold=0.05, n_iter=20)
        # library = [1, X, U, X*U, X²]
        assert Xi.shape[0] == 5
        # coef on X ≈ -2, on U ≈ 0.5; others ≈ 0
        assert abs(Xi[1, 0] + 2.0) < 0.1
        assert abs(Xi[2, 0] - 0.5) < 0.1
