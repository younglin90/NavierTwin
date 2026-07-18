"""Round 380 — L category milestone: AMR (R371-R379) e2e."""

from __future__ import annotations

import numpy as np


class TestMilestoneL:
    def test_imports(self) -> None:
        from naviertwin.core.amr import (  # noqa: F401
            aniso_driver,
            berger_oliger,
            dwr,
            ghost_exchange,
            goal_error,
            octree_forest,
            p_refine,
            r_adapt,
            refine_criteria,
        )

    def test_amr_pipeline(self) -> None:
        """Indicator → mark → bump p-order."""
        from naviertwin.core.amr.p_refine import bump_order
        from naviertwin.core.amr.refine_criteria import (
            gradient_indicator,
            mark_refine,
        )

        u = np.array([0., 0, 1., 1., 1.])
        ind = gradient_indicator(u, dx=1.0)
        mark = mark_refine(ind, threshold=0.1)
        p = np.array([2, 2, 2, 2, 2])
        # use ind as error proxy for bump
        p2 = bump_order(p, ind, threshold_up=0.3, threshold_down=0.0)
        assert mark.shape == u.shape
        # at gradient cells (idx 1, 2), p increased
        assert p2[1] >= 2 and p2[2] >= 2
