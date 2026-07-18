"""Round 276 — Space-time POD."""

from __future__ import annotations

import numpy as np


class TestSTPOD:
    def test_shapes(self) -> None:
        from naviertwin.core.dimensionality_reduction.linear.space_time_pod import (
            SpaceTimePOD,
        )

        rng = np.random.default_rng(0)
        X = rng.standard_normal((15, 40))
        stp = SpaceTimePOD(window=4, rank=5).fit(X)
        assert stp.modes.shape == (60, 5)
        assert stp.singular_values.shape == (5,)

    def test_low_rank_traveling(self) -> None:
        """traveling sine wave → space-time rank low."""
        from naviertwin.core.dimensionality_reduction.linear.space_time_pod import (
            SpaceTimePOD,
        )

        x = np.linspace(0, 2 * np.pi, 30)
        t = np.linspace(0, 4 * np.pi, 60)
        X = np.array([np.sin(x - tt) for tt in t]).T  # (30, 60)
        stp = SpaceTimePOD(window=5, rank=4).fit(X)
        # top 2 SVs >> rest
        s = stp.singular_values
        assert s[0] > 5 * s[-1]
