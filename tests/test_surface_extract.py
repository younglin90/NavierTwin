"""Round 306 — surface (boundary) extraction."""

from __future__ import annotations

import numpy as np


class TestSurfaceExtract:
    def test_single_tet(self) -> None:
        from naviertwin.core.tools.surface_extract import boundary_faces_tet

        tets = np.array([[0, 1, 2, 3]])
        faces = boundary_faces_tet(tets)
        assert faces.shape == (4, 3)

    def test_two_glued_tets(self) -> None:
        """Two tets sharing one face → 6 boundary faces."""
        from naviertwin.core.tools.surface_extract import boundary_faces_tet

        tets = np.array([[0, 1, 2, 3], [0, 1, 2, 4]])
        faces = boundary_faces_tet(tets)
        # 4+4 = 8 total, 1 shared face appears in both → 8 - 2 = 6
        assert faces.shape == (6, 3)
