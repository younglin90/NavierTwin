"""Round 259 — DTW."""

from __future__ import annotations

import numpy as np


class TestDTW:
    def test_identical_zero(self) -> None:
        from naviertwin.core.analysis.dtw import dtw_distance

        a = np.array([1.0, 2.0, 3.0, 4.0])
        assert dtw_distance(a, a) == 0.0

    def test_shifted(self) -> None:
        from naviertwin.core.analysis.dtw import dtw_distance

        a = np.array([1, 2, 3, 4], dtype=float)
        b = np.array([1, 1, 2, 3, 4], dtype=float)  # stutter on first element
        d = dtw_distance(a, b)
        assert d == 0.0  # warping aligns via repeated match

    def test_different(self) -> None:
        from naviertwin.core.analysis.dtw import dtw_distance

        a = np.array([1.0, 2.0, 3.0])
        b = np.array([10.0, 20.0, 30.0])
        assert dtw_distance(a, b) > dtw_distance(a, a + 0.5)

    def test_matrix(self) -> None:
        from naviertwin.core.analysis.dtw import dtw_matrix

        D = dtw_matrix(np.array([1.0, 2.0]), np.array([1.0, 2.0, 3.0]))
        assert D.shape == (3, 4)
        assert D[-1, -1] >= 0
