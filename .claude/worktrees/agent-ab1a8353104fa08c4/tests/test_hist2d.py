"""Round 436 — 2D histogram."""

from __future__ import annotations

import numpy as np


class TestHist2D:
    def test_shape(self) -> None:
        from naviertwin.core.visualization.hist2d import hist2d

        rng = np.random.default_rng(0)
        H, xe, ye = hist2d(rng.standard_normal(500), rng.standard_normal(500), bins=20)
        assert H.shape == (20, 20)
        assert H.sum() == 500

    def test_density(self) -> None:
        from naviertwin.core.visualization.hist2d import hist2d

        rng = np.random.default_rng(0)
        H, xe, ye = hist2d(rng.standard_normal(1000), rng.standard_normal(1000),
                            bins=20, density=True)
        # density integrates to 1 (approx)
        dxdy = (xe[1] - xe[0]) * (ye[1] - ye[0])
        assert abs(H.sum() * dxdy - 1.0) < 1e-9
