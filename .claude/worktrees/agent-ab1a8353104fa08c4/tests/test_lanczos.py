"""Round 265 — Lanczos."""

from __future__ import annotations

import numpy as np


class TestLanczos:
    def test_shapes_orthonormal(self) -> None:
        from naviertwin.core.linalg.lanczos import lanczos

        rng = np.random.default_rng(0)
        M = rng.standard_normal((6, 6))
        A = M + M.T
        Q, alpha, beta = lanczos(A, np.ones(6), k=4)
        assert Q.shape[1] <= 5
        assert alpha.shape[0] <= 4
        G = Q.T @ Q
        assert np.allclose(G, np.eye(Q.shape[1]), atol=1e-8)

    def test_ritz_values_extremes(self) -> None:
        from naviertwin.core.linalg.lanczos import ritz_values_sym

        A = np.diag([10.0, 5.0, 1.0, -2.0])
        rv = ritz_values_sym(A, np.ones(4), k=4)
        assert np.allclose(np.sort(rv), [-2.0, 1.0, 5.0, 10.0], atol=1e-6)
