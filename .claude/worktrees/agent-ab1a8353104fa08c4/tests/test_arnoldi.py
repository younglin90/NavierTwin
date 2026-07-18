"""Round 264 — Arnoldi + Ritz."""

from __future__ import annotations

import numpy as np


class TestArnoldi:
    def test_shapes_and_orthonormal(self) -> None:
        from naviertwin.core.linalg.arnoldi import arnoldi

        A = np.diag([5.0, 4.0, 3.0, 2.0, 1.0])
        Q, H = arnoldi(A, np.ones(5), k=3)
        assert Q.shape == (5, 4)
        assert H.shape == (4, 3)
        # orthonormal columns of Q
        G = Q.T @ Q
        assert np.allclose(G, np.eye(4), atol=1e-8)

    def test_ritz_values(self) -> None:
        from naviertwin.core.linalg.arnoldi import ritz_values

        A = np.diag([10.0, 5.0, 1.0])
        rv = ritz_values(A, np.ones(3), k=3)
        rv_sorted = sorted(rv.real)
        true_sorted = sorted([1.0, 5.0, 10.0])
        assert np.allclose(rv_sorted, true_sorted, atol=1e-6)
