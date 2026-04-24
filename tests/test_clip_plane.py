"""Round 304 — clip plane."""

from __future__ import annotations

import numpy as np


class TestClipPlane:
    def test_half_space(self) -> None:
        from naviertwin.core.tools.clip_plane import clip_points

        pts = np.array([[1., 0, 0], [-1., 0, 0], [2., 1, 0]])
        mask = clip_points(pts, n=np.array([1., 0, 0]), p=np.zeros(3))
        assert mask.tolist() == [True, False, True]

    def test_clip_triangles(self) -> None:
        from naviertwin.core.tools.clip_plane import clip_triangles

        pts = np.array([[1., 0, 0], [2., 1, 0], [-1., 0, 0]])
        tri = np.array([[0, 1, 2], [0, 1, 0]])
        kept = clip_triangles(
            pts, tri, n=np.array([1., 0, 0]), p=np.zeros(3),
        )
        # tri[0] has a vertex at x=-1 → dropped; tri[1] all on +side → kept
        assert kept.shape[0] == 1
        assert (kept[0] == [0, 1, 0]).all()
