"""Round 153 — GridIndex."""

from __future__ import annotations

import numpy as np


class TestGrid:
    def test_radius_matches_brute(self) -> None:
        from naviertwin.core.analysis.spatial_index import GridIndex

        rng = np.random.default_rng(0)
        pts = rng.uniform(-1, 1, size=(500, 3))
        idx = GridIndex(pts, cell_size=0.2)
        q = np.array([0.1, 0.0, -0.2])
        r = 0.3
        hits = sorted(idx.query_radius(q, r).tolist())
        brute = sorted(np.where(np.linalg.norm(pts - q, axis=1) <= r)[0].tolist())
        assert hits == brute

    def test_nearest(self) -> None:
        from naviertwin.core.analysis.spatial_index import GridIndex

        pts = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=float)
        idx = GridIndex(pts, cell_size=0.5)
        assert idx.nearest(np.array([0.9, 0, 0])) == 1
        assert idx.nearest(np.array([0, 0.9, 0])) == 2

    def test_2d(self) -> None:
        from naviertwin.core.analysis.spatial_index import GridIndex

        rng = np.random.default_rng(0)
        pts = rng.uniform(0, 1, size=(200, 2))
        idx = GridIndex(pts, cell_size=0.1)
        hits = idx.query_radius(np.array([0.5, 0.5]), 0.1)
        assert hits.size <= len(pts)
        # 모두 반경 안
        for i in hits:
            assert np.linalg.norm(pts[i] - np.array([0.5, 0.5])) <= 0.1 + 1e-9
