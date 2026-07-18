"""Round 252 — Poisson-disk."""

from __future__ import annotations

import numpy as np


class TestPoissonDisk:
    def test_min_distance(self) -> None:
        from naviertwin.core.sampling.poisson_disk import poisson_disk_2d

        r = 0.08
        pts = poisson_disk_2d(1.0, 1.0, r=r, seed=0)
        assert pts.shape[0] > 10
        # 모든 쌍 거리 >= r - tiny
        D = np.linalg.norm(pts[:, None, :] - pts[None, :, :], axis=2)
        np.fill_diagonal(D, np.inf)
        assert D.min() >= r - 1e-9

    def test_within_bounds(self) -> None:
        from naviertwin.core.sampling.poisson_disk import poisson_disk_2d

        pts = poisson_disk_2d(2.0, 1.0, r=0.15, seed=0)
        assert np.all(pts[:, 0] >= 0) and np.all(pts[:, 0] < 2.0)
        assert np.all(pts[:, 1] >= 0) and np.all(pts[:, 1] < 1.0)
