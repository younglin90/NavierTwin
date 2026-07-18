"""Round 339 — shape optimization."""

from __future__ import annotations

import numpy as np


class TestShapeOpt:
    def test_bezier_endpoints(self) -> None:
        from naviertwin.core.optimization.shape_opt import bezier_eval

        ctrl = np.array([[0., 0], [0.5, 1], [1., 0]])
        pts = bezier_eval(ctrl, t=np.array([0.0, 1.0]))
        assert np.allclose(pts[0], [0, 0])
        assert np.allclose(pts[1], [1, 0])

    def test_minimize_arc_length(self) -> None:
        """Minimize arc length of Bezier with fixed endpoints → straight line (mid → on segment)."""
        from naviertwin.core.optimization.shape_opt import (
            bezier_eval,
            optimize_bezier,
        )

        ctrl0 = np.array([[0.0, 0.0], [0.5, 1.0], [1.0, 0.0]])

        def length(c):
            t = np.linspace(0, 1, 30)
            p = bezier_eval(c, t)
            return float(np.sum(np.linalg.norm(np.diff(p, axis=0), axis=1)))

        ctrl = optimize_bezier(length, ctrl0, max_iter=80, lr=0.05)
        # mid control y should decrease (toward 0)
        assert ctrl[1, 1] < ctrl0[1, 1]
