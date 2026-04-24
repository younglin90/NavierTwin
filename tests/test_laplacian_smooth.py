"""Round 301 — Laplacian smoothing."""

from __future__ import annotations

import numpy as np


class TestLaplacianSmooth:
    def test_perturbed_line_relaxes(self) -> None:
        from naviertwin.core.tools.laplacian_smooth import laplacian_smooth

        # straight line with one displaced midpoint
        v = np.array([[0., 0.], [1., 0.5], [2., 0.]])
        edges = [(0, 1), (1, 2)]
        v2 = laplacian_smooth(v, edges, n_iter=10, alpha=0.5, fixed=[0, 2])
        # midpoint pulled toward y=0
        assert abs(v2[1, 1]) < abs(v[1, 1])

    def test_fixed_endpoints(self) -> None:
        from naviertwin.core.tools.laplacian_smooth import laplacian_smooth

        v = np.array([[0., 0.], [1., 1.], [2., 0.]])
        edges = [(0, 1), (1, 2)]
        v2 = laplacian_smooth(v, edges, n_iter=20, fixed=[0, 2])
        assert np.allclose(v2[0], v[0])
        assert np.allclose(v2[2], v[2])
