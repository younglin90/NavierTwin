"""Round 159 — Delaunay + P1 FEM."""

from __future__ import annotations

import numpy as np
import pytest


class TestDelaunay:
    def test_triangulate(self) -> None:
        pytest.importorskip("scipy")
        from naviertwin.core.tools.delaunay_2d import triangulate

        pts = np.array([[0, 0], [1, 0], [0, 1], [1, 1]], dtype=float)
        tri = triangulate(pts)
        assert tri["simplices"].shape[1] == 3
        assert tri["simplices"].shape[0] == 2

    def test_triangle_areas(self) -> None:
        pytest.importorskip("scipy")
        from naviertwin.core.tools.delaunay_2d import triangle_areas, triangulate

        pts = np.array([[0, 0], [1, 0], [0, 1], [1, 1]], dtype=float)
        t = triangulate(pts)
        a = triangle_areas(t["points"], t["simplices"])
        assert np.allclose(a.sum(), 1.0)

    def test_mass(self) -> None:
        pytest.importorskip("scipy")
        from naviertwin.core.tools.delaunay_2d import (
            lumped_mass_matrix,
            triangulate,
        )

        pts = np.array([[0, 0], [1, 0], [0, 1], [1, 1]], dtype=float)
        t = triangulate(pts)
        m = lumped_mass_matrix(t["points"], t["simplices"])
        assert m.shape == (4,)
        # 총합 = 전체 면적 = 1
        assert abs(m.sum() - 1.0) < 1e-12

    def test_stiffness(self) -> None:
        pytest.importorskip("scipy")
        from naviertwin.core.tools.delaunay_2d import (
            p1_stiffness_matrix,
            triangulate,
        )

        pts = np.array([[0, 0], [1, 0], [0, 1], [1, 1]], dtype=float)
        t = triangulate(pts)
        K = p1_stiffness_matrix(t["points"], t["simplices"])
        # K · 1 = 0 (상수 모드는 에너지 0)
        assert np.allclose(K @ np.ones(4), 0.0, atol=1e-12)
