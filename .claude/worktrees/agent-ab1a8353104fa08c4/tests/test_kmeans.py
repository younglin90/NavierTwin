"""Round 196 — k-means++."""

from __future__ import annotations

import numpy as np


class TestKMeans:
    def test_recovers_centers(self) -> None:
        from naviertwin.core.analysis.kmeans import kmeans

        rng = np.random.default_rng(0)
        c1 = rng.standard_normal((100, 2)) * 0.1
        c2 = rng.standard_normal((100, 2)) * 0.1 + np.array([5.0, 5.0])
        X = np.vstack([c1, c2])
        centers, labels = kmeans(X, k=2, seed=0)
        # 두 센터가 [0,0] 과 [5,5] 근방
        ds = sorted([np.linalg.norm(c) for c in centers])
        assert ds[0] < 0.5
        assert abs(ds[1] - np.sqrt(50)) < 0.5

    def test_inertia(self) -> None:
        from naviertwin.core.analysis.kmeans import inertia, kmeans

        rng = np.random.default_rng(0)
        X = rng.standard_normal((80, 3))
        centers, labels = kmeans(X, k=4, seed=0)
        inr = inertia(X, centers, labels)
        assert inr > 0

    def test_shapes(self) -> None:
        from naviertwin.core.analysis.kmeans import kmeans

        X = np.random.default_rng(0).standard_normal((20, 5))
        c, lbl = kmeans(X, k=3, seed=0)
        assert c.shape == (3, 5)
        assert lbl.shape == (20,)
