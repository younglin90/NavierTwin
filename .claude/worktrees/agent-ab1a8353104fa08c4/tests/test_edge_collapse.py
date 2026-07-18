"""Round 307 — edge collapse."""

from __future__ import annotations

import numpy as np


class TestEdgeCollapse:
    def test_one_collapse_reduces_vertex(self) -> None:
        from naviertwin.core.tools.edge_collapse import edge_collapse_once

        v = np.array([[0., 0], [0.1, 0], [1., 0], [0.5, 1]])
        tri = np.array([[0, 1, 3], [1, 2, 3]])
        v2, tri2 = edge_collapse_once(v, tri)
        assert len(v2) < len(v)
        assert tri2.shape[1] == 3

    def test_indices_valid(self) -> None:
        from naviertwin.core.tools.edge_collapse import edge_collapse_once

        v = np.array([[0., 0], [0.1, 0.05], [1., 0], [0.5, 1], [0.5, -0.5]])
        tri = np.array([[0, 1, 3], [1, 2, 3], [0, 1, 4]])
        v2, tri2 = edge_collapse_once(v, tri)
        assert tri2.max() < len(v2)
        assert tri2.min() >= 0
