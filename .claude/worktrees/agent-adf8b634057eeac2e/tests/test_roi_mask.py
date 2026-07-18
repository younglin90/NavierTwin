"""Round 123 — ROI 마스크."""

from __future__ import annotations

import numpy as np
import pytest


class TestROI:
    def test_box(self) -> None:
        from naviertwin.core.analysis.roi_mask import box_mask

        pts = np.array([[0, 0, 0], [0.5, 0.5, 0.5], [2, 2, 2]], dtype=float)
        m = box_mask(pts, (0, 0, 0, 1, 1, 1))
        assert m.tolist() == [True, True, False]

    def test_sphere(self) -> None:
        from naviertwin.core.analysis.roi_mask import sphere_mask

        pts = np.array([[0, 0, 0], [1, 0, 0], [2, 0, 0]], dtype=float)
        m = sphere_mask(pts, center=(0, 0, 0), radius=1.5)
        assert m.tolist() == [True, True, False]

    def test_cylinder(self) -> None:
        from naviertwin.core.analysis.roi_mask import cylinder_mask

        pts = np.array([
            [0.5, 0, 0],     # 축 위
            [0.5, 0.1, 0],   # 내부
            [0.5, 2, 0],     # 반지름 밖
            [2.0, 0, 0],     # 끝 밖
        ], dtype=float)
        m = cylinder_mask(pts, start=(0, 0, 0), end=(1, 0, 0), radius=0.5)
        assert m.tolist() == [True, True, False, False]

    def test_plane(self) -> None:
        from naviertwin.core.analysis.roi_mask import plane_half_space

        pts = np.array([[0, 0, 1], [0, 0, -1]], dtype=float)
        m = plane_half_space(pts, origin=(0, 0, 0), normal=(0, 0, 1))
        assert m.tolist() == [True, False]

    def test_predicate(self) -> None:
        from naviertwin.core.analysis.roi_mask import predicate_mask

        pts = np.array([[0, 0, 0], [3, 4, 0]], dtype=float)
        m = predicate_mask(pts, lambda p: np.linalg.norm(p, axis=1) > 2)
        assert m.tolist() == [False, True]

    def test_cylinder_invalid(self) -> None:
        from naviertwin.core.analysis.roi_mask import cylinder_mask

        with pytest.raises(ValueError):
            cylinder_mask(np.zeros((1, 3)), (0, 0, 0), (0, 0, 0), 1.0)
