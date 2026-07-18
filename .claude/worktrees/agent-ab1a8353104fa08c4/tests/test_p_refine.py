"""Round 371 — p-refinement."""

from __future__ import annotations

import numpy as np


class TestPRefine:
    def test_bump(self) -> None:
        from naviertwin.core.amr.p_refine import bump_order

        p = np.array([2, 2, 2, 2])
        e = np.array([0.5, 0.5, 0.001, 0.001])
        p2 = bump_order(p, e, threshold_up=0.1, threshold_down=0.01)
        assert (p2 == [3, 3, 1, 1]).all()

    def test_clip_bounds(self) -> None:
        from naviertwin.core.amr.p_refine import bump_order

        p = np.array([6, 1])
        e = np.array([10.0, 0.0])
        p2 = bump_order(p, e, p_min=1, p_max=6)
        assert p2.tolist() == [6, 1]
