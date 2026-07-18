"""Round 165 — Pareto."""

from __future__ import annotations

import numpy as np


class TestPareto:
    def test_mask(self) -> None:
        from naviertwin.core.optimization.pareto import pareto_mask

        F = np.array([[1, 5], [2, 3], [3, 4], [5, 1]], dtype=float)
        m = pareto_mask(F)
        assert m.tolist() == [True, True, False, True]

    def test_dominated_removed(self) -> None:
        from naviertwin.core.optimization.pareto import pareto_mask

        F = np.array([[1, 1], [2, 2], [3, 3]], dtype=float)
        m = pareto_mask(F)
        # 첫번째가 나머지 지배
        assert m.tolist() == [True, False, False]

    def test_nondominated_sort(self) -> None:
        from naviertwin.core.optimization.pareto import nondominated_sort

        F = np.array([[1, 5], [2, 3], [3, 4], [5, 1], [2, 4]], dtype=float)
        fronts = nondominated_sort(F)
        assert len(fronts) >= 1
        # 첫 front 가 Pareto optimal
        assert set(fronts[0].tolist()).issubset({0, 1, 3})

    def test_hypervolume_2d(self) -> None:
        from naviertwin.core.optimization.pareto import hypervolume_2d

        F = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=float)
        hv = hypervolume_2d(F, ref=(2.0, 2.0))
        # 두 점으로 만들어지는 계단: (2-0)*(2-0) - 내부 사각형 = 3 (0 → 1 Δx=1, y_prev=2→1)
        assert hv == 3.0
