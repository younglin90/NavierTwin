"""Round 310 — E category milestone: mesh tools (R301-R309) e2e."""

from __future__ import annotations

import numpy as np


class TestMilestoneE:
    def test_imports(self) -> None:
        from naviertwin.core.tools import (  # noqa: F401
            aniso_metric,
            bl_orthogonal,
            clip_plane,
            edge_collapse,
            laplacian_smooth,
            marching_cubes_lite,
            stream_seeds,
            surface_extract,
            winslow,
        )

    def test_bbox_seed_streamline_smoke(self) -> None:
        """uniform seeds inside bbox passes clip plane."""
        from naviertwin.core.tools.clip_plane import clip_points
        from naviertwin.core.tools.stream_seeds import uniform_seeds

        seeds_2d = uniform_seeds(((0, 0), (1, 1)), n=20)
        # extend to 3D
        seeds = np.column_stack([seeds_2d, np.zeros(20)])
        mask = clip_points(seeds, n=np.array([0., 0, 1]), p=np.zeros(3))
        # all on plane (= 0) → not strictly > 0
        assert not mask.any()

    def test_smooth_then_metric(self) -> None:
        """Smooth mesh → compute metric edge length."""
        from naviertwin.core.tools.aniso_metric import edge_length_metric
        from naviertwin.core.tools.laplacian_smooth import laplacian_smooth

        v = np.array([[0., 0], [1., 0.3], [2., 0]])
        v2 = laplacian_smooth(v, [(0, 1), (1, 2)], n_iter=10, fixed=[0, 2])
        L = edge_length_metric(np.eye(2), np.eye(2), v2[0], v2[2])
        assert np.isclose(L, 2.0)
