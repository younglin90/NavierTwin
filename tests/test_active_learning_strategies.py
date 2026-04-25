"""Round 517 — active learning query strategies."""

from __future__ import annotations

import numpy as np


class TestAL:
    def test_uncertainty(self) -> None:
        from naviertwin.utils.active_learning import uncertainty_query

        probs = np.array([[0.9, 0.1], [0.55, 0.45], [0.7, 0.3]])
        idx = uncertainty_query(probs, k=1)
        assert idx[0] == 1

    def test_margin(self) -> None:
        from naviertwin.utils.active_learning import margin_query

        probs = np.array([[0.9, 0.1], [0.55, 0.45], [0.7, 0.3]])
        idx = margin_query(probs, k=1)
        assert idx[0] == 1

    def test_random(self) -> None:
        from naviertwin.utils.active_learning import random_query

        idx = random_query(n=20, k=5, seed=0)
        assert len(idx) == 5
