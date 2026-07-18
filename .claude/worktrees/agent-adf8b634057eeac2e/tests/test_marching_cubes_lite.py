"""Round 303 — marching squares."""

from __future__ import annotations

import numpy as np


class TestMS:
    def test_circle_contour(self) -> None:
        from naviertwin.core.tools.marching_cubes_lite import marching_squares

        n = 41
        x = np.linspace(-1, 1, n)
        X, Y = np.meshgrid(x, x, indexing="ij")
        f = X * X + Y * Y
        segs = marching_squares(f, level=0.5)
        assert len(segs) > 30  # closed contour ≈ many segments

    def test_no_crossing_empty(self) -> None:
        from naviertwin.core.tools.marching_cubes_lite import marching_squares

        f = np.ones((5, 5))
        segs = marching_squares(f, level=0.5)
        assert len(segs) == 0

    def test_segments_within_bounds(self) -> None:
        from naviertwin.core.tools.marching_cubes_lite import marching_squares

        n = 10
        x = np.linspace(-1, 1, n)
        X, Y = np.meshgrid(x, x, indexing="ij")
        f = X
        segs = marching_squares(f, level=0.0)
        for (a, b) in segs:
            assert 0 <= a[0] <= n - 1
            assert 0 <= a[1] <= n - 1
            assert 0 <= b[0] <= n - 1
