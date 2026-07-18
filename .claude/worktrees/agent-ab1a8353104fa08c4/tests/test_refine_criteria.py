"""Round 379 — refinement criteria."""

from __future__ import annotations

import numpy as np


class TestRefineCriteria:
    def test_gradient_step(self) -> None:
        from naviertwin.core.amr.refine_criteria import gradient_indicator

        u = np.array([0., 0, 1., 1., 1.])
        g = gradient_indicator(u, dx=1.0)
        assert g[0] == 0
        assert g[2] > 0  # at the jump

    def test_curvature_peak(self) -> None:
        from naviertwin.core.amr.refine_criteria import curvature_indicator

        u = np.array([0., 0., 1., 0., 0.])
        c = curvature_indicator(u, dx=1.0)
        # max curvature at peak
        assert c.argmax() == 2

    def test_mark(self) -> None:
        from naviertwin.core.amr.refine_criteria import mark_refine

        ind = np.array([0.05, 0.2, 0.5])
        assert mark_refine(ind, threshold=0.1).tolist() == [False, True, True]
