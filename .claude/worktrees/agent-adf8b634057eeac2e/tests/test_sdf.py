"""Round 381 — SDF primitives."""

from __future__ import annotations

import numpy as np


class TestSDF:
    def test_sphere(self) -> None:
        from naviertwin.core.geometry.sdf import sdf_sphere

        # inside center, sdf = -r
        assert sdf_sphere(np.zeros(3), center=np.zeros(3), r=1.0) == -1.0
        # on surface
        assert np.isclose(sdf_sphere(np.array([1, 0, 0]), center=np.zeros(3), r=1.0), 0.0)
        # outside
        assert sdf_sphere(np.array([2, 0, 0]), center=np.zeros(3), r=1.0) == 1.0

    def test_box(self) -> None:
        from naviertwin.core.geometry.sdf import sdf_box

        # inside box
        d = sdf_box(np.zeros(3), center=np.zeros(3), half_size=np.array([1, 1, 1.]))
        assert d < 0
        # outside
        d = sdf_box(np.array([2, 2, 2.]), center=np.zeros(3),
                     half_size=np.array([1, 1, 1.]))
        assert d > 0

    def test_plane(self) -> None:
        from naviertwin.core.geometry.sdf import sdf_plane

        d = sdf_plane(np.array([0, 1, 0.]), normal=np.array([0, 1, 0.]), offset=0)
        assert d == 1.0
