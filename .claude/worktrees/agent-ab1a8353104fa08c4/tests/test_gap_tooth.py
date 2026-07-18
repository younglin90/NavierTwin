"""Round 468 — gap-tooth."""

from __future__ import annotations

import numpy as np


class TestGapTooth:
    def test_fill(self) -> None:
        from naviertwin.core.multiscale.gap_tooth import gap_tooth_fill

        u = gap_tooth_fill(
            patch_centers=np.array([0, 4, 8]),
            patch_vals=np.array([0.0, 4.0, 8.0]),
            n_full=9,
        )
        # linear from 0..8
        assert np.allclose(u, np.arange(9.0))

    def test_shape(self) -> None:
        from naviertwin.core.multiscale.gap_tooth import gap_tooth_fill

        u = gap_tooth_fill(np.array([2, 5]), np.array([1.0, 2.0]), n_full=8)
        assert u.shape == (8,)
