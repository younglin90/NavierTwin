"""Round 188 — 메쉬 품질."""

from __future__ import annotations

import numpy as np
import pytest


class TestMeshQuality:
    def test_equilateral_ar_one(self) -> None:
        from naviertwin.core.tools.mesh_quality import aspect_ratio_triangle

        tri = np.array([[0, 0], [1, 0], [0.5, np.sqrt(3) / 2]], dtype=float)
        ar = aspect_ratio_triangle(tri)
        assert abs(ar - 1.0) < 1e-9

    def test_thin_triangle_high_ar(self) -> None:
        from naviertwin.core.tools.mesh_quality import aspect_ratio_triangle

        tri = np.array([[0, 0], [1, 0], [0.5, 0.01]], dtype=float)
        ar = aspect_ratio_triangle(tri)
        assert ar > 10

    def test_min_angle_equilateral(self) -> None:
        from naviertwin.core.tools.mesh_quality import min_angle_deg

        tri = np.array([[0, 0], [1, 0], [0.5, np.sqrt(3) / 2]], dtype=float)
        assert abs(min_angle_deg(tri) - 60.0) < 1e-6

    def test_skewness_equilateral(self) -> None:
        from naviertwin.core.tools.mesh_quality import skewness_equilateral

        tri = np.array([[0, 0], [1, 0], [0.5, np.sqrt(3) / 2]], dtype=float)
        assert skewness_equilateral(tri) < 1e-6

    def test_report(self) -> None:
        pytest.importorskip("scipy")
        from naviertwin.core.tools.delaunay_2d import triangulate
        from naviertwin.core.tools.mesh_quality import mesh_quality_report

        pts = np.array([[0, 0], [1, 0], [0, 1], [1, 1]], dtype=float)
        t = triangulate(pts)
        rep = mesh_quality_report(t["points"], t["simplices"])
        assert "mean_aspect" in rep
        assert rep["min_angle_deg"] > 0
