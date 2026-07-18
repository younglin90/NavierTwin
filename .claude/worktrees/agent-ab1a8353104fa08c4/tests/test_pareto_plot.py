"""Round 435 — Pareto front."""

from __future__ import annotations

import numpy as np


class TestPareto:
    def test_basic(self) -> None:
        from naviertwin.core.visualization.pareto_plot import pareto_front

        obj = np.array([[1, 5], [2, 3], [3, 4], [4, 1]])
        mask = pareto_front(obj)
        # (3, 4) dominated by (2, 3); rest non-dominated
        assert mask.tolist() == [True, True, False, True]

    def test_single_point(self) -> None:
        from naviertwin.core.visualization.pareto_plot import pareto_front

        mask = pareto_front(np.array([[1.0, 1.0]]))
        assert mask.tolist() == [True]
