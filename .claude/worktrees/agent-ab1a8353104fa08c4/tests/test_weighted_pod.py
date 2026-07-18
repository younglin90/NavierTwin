"""Round 215 — weighted POD / cPOD."""

from __future__ import annotations

import numpy as np


class TestWPOD:
    def test_identity_weight_matches_standard(self) -> None:
        from naviertwin.core.dimensionality_reduction.linear.snapshot_pod_scratch import (
            snapshot_pod,
        )
        from naviertwin.core.dimensionality_reduction.linear.weighted_pod import (
            weighted_pod,
        )

        rng = np.random.default_rng(0)
        X = rng.standard_normal((30, 12))
        _, sv_a, _ = snapshot_pod(X, k=5)
        _, sv_b = weighted_pod(X, None, k=5)
        assert np.allclose(sorted(sv_a), sorted(sv_b), atol=1e-8)

    def test_weighted(self) -> None:
        from naviertwin.core.dimensionality_reduction.linear.weighted_pod import (
            weighted_pod,
        )

        rng = np.random.default_rng(0)
        X = rng.standard_normal((20, 8))
        W = np.diag(np.linspace(0.5, 2.0, 20))
        modes, sv = weighted_pod(X, W, k=3)
        assert modes.shape == (20, 3)
        assert sv.shape == (3,)

    def test_compressed(self) -> None:
        from naviertwin.core.dimensionality_reduction.linear.weighted_pod import (
            compressed_pod,
        )

        rng = np.random.default_rng(0)
        X = rng.standard_normal((50, 10))
        modes, sv = compressed_pod(X, k=3, compress=20, seed=0)
        assert modes.shape == (50, 3)
        assert sv.shape == (3,)
        # mode 들은 정규화
        assert np.allclose(np.linalg.norm(modes, axis=0), 1.0, atol=1e-8)
