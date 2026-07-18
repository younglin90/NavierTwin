"""Round 509 — Sobol."""

from __future__ import annotations

import numpy as np


class TestSobol:
    def test_dominant_dim(self) -> None:
        from naviertwin.core.verification.sobol import sobol_indices

        rng = np.random.default_rng(0)
        # y = X0 + 0.01 X1 → S0 >> S1
        def model(X):
            return X[:, 0] + 0.01 * X[:, 1]

        S, ST = sobol_indices(model, n_dim=2, n_samples=4000, rng=rng)
        assert S[0] > S[1]
        assert ST[0] > ST[1]
