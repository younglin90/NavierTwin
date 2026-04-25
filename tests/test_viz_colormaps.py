"""Round 432 — visualization colormaps."""

from __future__ import annotations

import numpy as np


class TestVizCmap:
    def test_viridis_shape(self) -> None:
        from naviertwin.core.visualization.colormaps import apply_cmap

        rgb = apply_cmap(np.linspace(0, 1, 10), name="viridis")
        assert rgb.shape == (10, 3)
        assert ((rgb >= 0) & (rgb <= 1)).all()

    def test_gray(self) -> None:
        from naviertwin.core.visualization.colormaps import apply_cmap

        rgb = apply_cmap(np.array([0.5]), name="gray")
        assert np.allclose(rgb[0], [0.5, 0.5, 0.5])

    def test_jet_endpoints(self) -> None:
        from naviertwin.core.visualization.colormaps import apply_cmap

        rgb = apply_cmap(np.array([0.0, 1.0]), name="jet")
        assert rgb[0, 2] > rgb[0, 0]
        assert rgb[1, 0] > rgb[1, 2]
