"""Round 409 — reduced barycentric."""

from __future__ import annotations

import numpy as np


class TestRB:
    def test_sum_to_one(self) -> None:
        from naviertwin.core.dimensionality_reduction.nonlinear.reduced_barycentric import (
            barycentric_weights,
        )

        V = np.array([[0., 0], [1., 0], [0., 1.]])
        w = barycentric_weights(np.array([0.3, 0.3]), V)
        assert abs(w.sum() - 1.0) < 1e-6
        assert (w >= 0).all()

    def test_corner(self) -> None:
        from naviertwin.core.dimensionality_reduction.nonlinear.reduced_barycentric import (
            barycentric_weights,
        )

        V = np.array([[0., 0], [1., 0], [0., 1.]])
        # at first anchor → w ≈ [1, 0, 0]
        w = barycentric_weights(np.array([0.0, 0.0]), V)
        assert w[0] > 0.9
