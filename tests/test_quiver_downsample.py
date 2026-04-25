"""Round 438 — quiver downsample."""

from __future__ import annotations

import numpy as np


class TestQuiverDS:
    def test_count(self) -> None:
        from naviertwin.core.visualization.quiver_downsample import downsample_3d

        X, Y, Z = np.mgrid[:10, :10, :10]
        pts = np.stack([X, Y, Z], axis=-1).astype(float)
        vec = pts.copy()
        p_ds, v_ds = downsample_3d(pts, vec, stride=2)
        # ceil(10/2) = 5, → 5³ = 125
        assert p_ds.shape == (125, 3)
        assert v_ds.shape == (125, 3)
