"""Round 158 — KDTree wrapper."""

from __future__ import annotations

import numpy as np
import pytest


class TestKDTree:
    def test_knn(self) -> None:
        pytest.importorskip("scipy")
        from naviertwin.core.analysis.kdtree_wrapper import KDTreeIndex

        pts = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=float)
        kd = KDTreeIndex(pts)
        d, idx = kd.knn(np.array([[0.1, 0.0, 0.0]]), k=1)
        assert idx[0, 0] == 0

    def test_radius(self) -> None:
        pytest.importorskip("scipy")
        from naviertwin.core.analysis.kdtree_wrapper import KDTreeIndex

        rng = np.random.default_rng(0)
        pts = rng.standard_normal((500, 2))
        kd = KDTreeIndex(pts)
        q = np.array([0.0, 0.0])
        idx = kd.radius(q, 0.2)
        # brute 비교
        brute = np.where(np.linalg.norm(pts - q, axis=1) <= 0.2)[0]
        assert sorted(idx.tolist()) == sorted(brute.tolist())

    def test_count_within(self) -> None:
        pytest.importorskip("scipy")
        from naviertwin.core.analysis.kdtree_wrapper import KDTreeIndex

        pts = np.random.default_rng(0).standard_normal((200, 3))
        kd = KDTreeIndex(pts)
        c = kd.count_within(np.zeros(3), 0.5)
        assert c >= 0
