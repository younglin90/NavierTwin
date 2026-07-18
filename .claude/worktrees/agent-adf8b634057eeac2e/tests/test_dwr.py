"""Round 376 — DWR."""

from __future__ import annotations

import numpy as np


class TestDWR:
    def test_zero_when_dual_match(self) -> None:
        from naviertwin.core.amr.dwr import dwr_indicator

        R = np.array([1.0, 2.0])
        z = np.array([0.5, 0.5])
        eta = dwr_indicator(R, z, z)
        assert (eta == 0).all()

    def test_indicator(self) -> None:
        from naviertwin.core.amr.dwr import dwr_indicator

        eta = dwr_indicator(
            np.array([0.5]),
            np.array([1.0]),
            np.array([0.6]),
        )
        assert np.isclose(eta[0], 0.5 * 0.4)
