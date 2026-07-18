"""Round 187 — barycentric."""

from __future__ import annotations

import numpy as np
import pytest


class TestBary:
    def test_sum_one(self) -> None:
        from naviertwin.core.analysis.barycentric import barycentric_2d

        tri = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
        p = np.array([0.2, 0.3])
        bc = barycentric_2d(tri, p)
        assert abs(bc.sum() - 1.0) < 1e-12
        assert np.all(bc >= 0)

    def test_vertex_returns_one_hot(self) -> None:
        from naviertwin.core.analysis.barycentric import barycentric_2d

        tri = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
        bc = barycentric_2d(tri, tri[1])
        assert abs(bc[1] - 1.0) < 1e-12
        assert abs(bc[0]) < 1e-12

    def test_triangle_interp(self) -> None:
        from naviertwin.core.analysis.barycentric import triangle_interp

        tri = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
        vals = np.array([0.0, 10.0, 20.0])
        # center (1/3, 1/3) → (0 + 10 + 20)/3
        val = triangle_interp(tri, vals, np.array([1 / 3, 1 / 3]))
        assert abs(val - 30 / 3) < 1e-10

    def test_locate(self) -> None:
        pytest.importorskip("scipy")
        from naviertwin.core.analysis.barycentric import locate_triangle
        from naviertwin.core.tools.delaunay_2d import triangulate

        pts = np.array([[0, 0], [1, 0], [0, 1], [1, 1]], dtype=float)
        t = triangulate(pts)
        idx = locate_triangle(t["points"], t["simplices"], np.array([0.25, 0.25]))
        assert idx >= 0
