"""Round 388 — CSG."""

from __future__ import annotations

import numpy as np


class TestCSG:
    def test_union(self) -> None:
        from naviertwin.core.geometry.csg import csg_union

        a = np.array([1.0, -1.0])
        b = np.array([-1.0, 1.0])
        assert (csg_union(a, b) == [-1.0, -1.0]).all()

    def test_intersect(self) -> None:
        from naviertwin.core.geometry.csg import csg_intersect

        a = np.array([1.0, -1.0])
        b = np.array([-1.0, 1.0])
        assert (csg_intersect(a, b) == [1.0, 1.0]).all()

    def test_diff(self) -> None:
        from naviertwin.core.geometry.csg import csg_diff

        a = np.array([-1.0])  # inside A
        b = np.array([-1.0])  # inside B → should be removed
        assert csg_diff(a, b)[0] == 1.0
