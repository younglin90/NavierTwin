"""Round 298 — Okubo-Weiss."""

from __future__ import annotations

import numpy as np


class TestOkuboWeiss:
    def test_pure_rotation_negative(self) -> None:
        from naviertwin.core.analysis.okubo_weiss import okubo_weiss

        grad = np.zeros((1, 2, 2))
        grad[0] = [[0, -1], [1, 0]]
        W = okubo_weiss(grad)
        assert W[0] < 0

    def test_pure_strain_positive(self) -> None:
        from naviertwin.core.analysis.okubo_weiss import okubo_weiss

        grad = np.zeros((1, 2, 2))
        grad[0] = [[1, 0], [0, -1]]  # extension/compression
        W = okubo_weiss(grad)
        assert W[0] > 0

    def test_shape(self) -> None:
        from naviertwin.core.analysis.okubo_weiss import okubo_weiss

        rng = np.random.default_rng(0)
        grad = rng.standard_normal((5, 7, 2, 2))
        W = okubo_weiss(grad)
        assert W.shape == (5, 7)
