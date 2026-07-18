"""Round 195 — DBSCAN."""

from __future__ import annotations

import numpy as np


class TestDBSCAN:
    def test_two_clusters(self) -> None:
        from naviertwin.core.analysis.dbscan import dbscan, n_clusters

        rng = np.random.default_rng(0)
        c1 = rng.standard_normal((50, 2)) * 0.1
        c2 = rng.standard_normal((50, 2)) * 0.1 + np.array([5.0, 5.0])
        X = np.vstack([c1, c2])
        labels = dbscan(X, eps=0.3, min_samples=3)
        assert n_clusters(labels) >= 2

    def test_noise_isolated(self) -> None:
        from naviertwin.core.analysis.dbscan import dbscan

        X = np.array([
            [0, 0], [0.1, 0], [0.2, 0.1], [0.05, 0.05], [0, 0.1],
            [10, 10],  # noise
        ], dtype=float)
        labels = dbscan(X, eps=0.3, min_samples=3)
        assert labels[-1] == -1

    def test_all_noise(self) -> None:
        from naviertwin.core.analysis.dbscan import dbscan, n_clusters

        X = np.arange(10).reshape(-1, 1).astype(float) * 100
        labels = dbscan(X, eps=0.1, min_samples=3)
        assert n_clusters(labels) == 0
