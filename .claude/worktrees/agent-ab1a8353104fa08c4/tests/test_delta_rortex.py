"""Round 299 — Δ-criterion / Rortex."""

from __future__ import annotations

import numpy as np


class TestDeltaRortex:
    def test_delta_rotation_positive(self) -> None:
        from naviertwin.core.analysis.delta_rortex import delta_criterion

        grad = np.zeros((1, 3, 3))
        grad[0] = [[0, -1, 0], [1, 0, 0], [0, 0, 0]]
        D = delta_criterion(grad)
        assert D[0] > 0

    def test_rortex_rotation(self) -> None:
        from naviertwin.core.analysis.delta_rortex import rortex_field

        grad = np.zeros((1, 3, 3))
        grad[0] = [[0, -1, 0], [1, 0, 0], [0, 0, 0]]
        R = rortex_field(grad)
        assert np.isclose(R[0], 2.0)

    def test_zero_field(self) -> None:
        from naviertwin.core.analysis.delta_rortex import (
            delta_criterion,
            rortex_field,
        )

        grad = np.zeros((3, 3, 3))
        assert (delta_criterion(grad) == 0).all()
        assert (rortex_field(grad) == 0).all()
