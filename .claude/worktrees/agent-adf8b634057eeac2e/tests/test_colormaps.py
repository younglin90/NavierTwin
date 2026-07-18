"""Round 151 — 색맵."""

from __future__ import annotations

import numpy as np
import pytest


class TestColormap:
    def test_list(self) -> None:
        from naviertwin.gui.colormaps import available_colormaps

        maps = available_colormaps()
        assert "viridis" in maps
        assert "jet" in maps

    @pytest.mark.parametrize("name", ["viridis", "plasma", "turbo", "coolwarm", "jet"])
    def test_shapes_and_range(self, name: str) -> None:
        from naviertwin.gui.colormaps import apply_colormap

        v = np.linspace(0, 1, 64)
        rgb = apply_colormap(v, name)
        assert rgb.shape == (64, 3)
        assert rgb.min() >= 0.0
        assert rgb.max() <= 1.0

    def test_invalid(self) -> None:
        from naviertwin.gui.colormaps import apply_colormap

        with pytest.raises(ValueError):
            apply_colormap(np.zeros(3), "bogus")

    def test_clip(self) -> None:
        from naviertwin.gui.colormaps import apply_colormap

        rgb = apply_colormap(np.array([-10.0, 0.0, 1.0, 100.0]), "viridis", vmin=0, vmax=1)
        assert rgb[0].tolist() == pytest.approx(rgb[1].tolist())
        assert rgb[-1].tolist() == pytest.approx(rgb[-2].tolist())
