"""Round 207 — snapshot POD scratch."""

from __future__ import annotations

import numpy as np


class TestSnapPOD:
    def test_reconstruction_rank_k(self) -> None:
        from naviertwin.core.dimensionality_reduction.linear.snapshot_pod_scratch import (
            reconstruct,
            snapshot_pod,
        )

        rng = np.random.default_rng(0)
        L = rng.standard_normal((50, 3))
        R = rng.standard_normal((3, 20))
        X = L @ R  # rank 3
        mean = X.mean(axis=1, keepdims=True)
        modes, sv, coeffs = snapshot_pod(X, k=3)
        X_hat = reconstruct(modes, coeffs, mean=mean)
        assert np.linalg.norm(X - X_hat) / np.linalg.norm(X) < 1e-8

    def test_sv_matches_svd(self) -> None:
        from naviertwin.core.dimensionality_reduction.linear.snapshot_pod_scratch import (
            snapshot_pod,
        )

        rng = np.random.default_rng(0)
        X = rng.standard_normal((40, 15))
        Xc = X - X.mean(axis=1, keepdims=True)
        _, svd_s, _ = np.linalg.svd(Xc, full_matrices=False)
        _, sv, _ = snapshot_pod(X, k=5)
        assert np.allclose(np.sort(sv), np.sort(svd_s[:5]), atol=1e-8)
