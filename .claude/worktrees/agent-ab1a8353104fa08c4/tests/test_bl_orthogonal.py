"""Round 309 — boundary layer grid."""

from __future__ import annotations

import numpy as np


class TestBL:
    def test_shape(self) -> None:
        from naviertwin.core.tools.bl_orthogonal import bl_grid

        wall = np.array([[0., 0], [1., 0]])
        n = np.array([[0., 1], [0., 1]])
        g = bl_grid(wall, n, n_layers=5, first=0.01, growth=1.2)
        assert g.shape == (2, 5, 2)

    def test_first_layer_thickness(self) -> None:
        from naviertwin.core.tools.bl_orthogonal import bl_grid

        wall = np.array([[0., 0]])
        n = np.array([[0., 1]])
        g = bl_grid(wall, n, n_layers=3, first=0.01, growth=2.0)
        # y values: 0.01, 0.01+0.02=0.03, 0.03+0.04=0.07
        assert np.isclose(g[0, 0, 1], 0.01)
        assert np.isclose(g[0, 1, 1], 0.03)
        assert np.isclose(g[0, 2, 1], 0.07)

    def test_orthogonal_to_wall(self) -> None:
        from naviertwin.core.tools.bl_orthogonal import bl_grid

        wall = np.array([[0., 0]])
        n = np.array([[1., 0]])
        g = bl_grid(wall, n, n_layers=3, first=0.1, growth=1.0)
        # all points along x-axis
        assert np.allclose(g[0, :, 1], 0)
