"""Round 108 — 메쉬 통계."""

from __future__ import annotations

import numpy as np
import pytest


class TestMeshStats:
    def test_bounding_box(self) -> None:
        from naviertwin.core.tools.mesh_stats import bounding_box

        pts = np.array([[0, 0, 0], [1, 2, 3], [4, 5, 6]], dtype=float)
        b = bounding_box(pts)
        assert b["xmin"] == 0 and b["xmax"] == 4
        assert b["dy"] == 5
        assert b["diagonal"] > 0

    def test_bbox_invalid(self) -> None:
        from naviertwin.core.tools.mesh_stats import bounding_box

        with pytest.raises(ValueError):
            bounding_box(np.zeros((5, 2)))

    def test_edge_length_explicit(self) -> None:
        from naviertwin.core.tools.mesh_stats import edge_length_stats

        pts = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0]], dtype=float)
        edges = np.array([[0, 1], [1, 2], [0, 2]])
        s = edge_length_stats(pts, edges)
        assert s["min"] == pytest.approx(1.0)
        assert s["max"] == pytest.approx(np.sqrt(2.0))

    def test_pyvista_integration(self) -> None:
        pytest.importorskip("pyvista")
        import pyvista as pv

        from naviertwin.core.tools.mesh_stats import mesh_stats

        cube = pv.Cube()
        s = mesh_stats(cube)
        assert s["n_points"] > 0
        assert s["n_cells"] > 0
        assert s["bbox"]["dx"] > 0
