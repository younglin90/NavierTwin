"""Round 164 — Active learning."""

from __future__ import annotations

import numpy as np


class TestAL:
    def test_top_variance(self) -> None:
        from naviertwin.core.active_learning.query import top_variance_query

        v = np.array([0.1, 0.3, 0.8, 0.2, 0.05])
        idx = top_variance_query(v, k=3)
        assert idx.tolist() == [2, 1, 3]

    def test_greedy_maxmin(self) -> None:
        from naviertwin.core.active_learning.query import greedy_maxmin

        cand = np.array([[0, 0], [10, 10], [1, 1], [5, 5]], dtype=float)
        existing = np.array([[0, 0]], dtype=float)
        idx = greedy_maxmin(cand, existing, k=1)
        # 가장 먼 = (10,10)
        assert idx[0] == 1

    def test_ei(self) -> None:
        from naviertwin.core.active_learning.query import expected_improvement

        mu = np.array([0.5, 0.9, 1.2])
        sigma = np.array([0.3, 0.1, 0.5])
        ei = expected_improvement(mu, sigma, y_best=1.0)
        assert ei.shape == (3,)
        assert np.all(ei >= 0)
        # mu < y_best 일 때 EI > 0
        assert ei[0] > 0
