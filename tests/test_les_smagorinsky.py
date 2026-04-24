"""Round 293 — LES Smagorinsky."""

from __future__ import annotations

import numpy as np


class TestLES:
    def test_nu_sgs_formula(self) -> None:
        from naviertwin.core.analysis.les_smagorinsky import (
            CS_DEFAULT,
            smagorinsky_nu_sgs,
        )

        S = np.array([1.0, 4.0, 0.0])
        nu = smagorinsky_nu_sgs(S, dx=0.1)
        expected = (CS_DEFAULT * 0.1) ** 2 * S
        assert np.allclose(nu, expected)

    def test_filter_smooths(self) -> None:
        from naviertwin.core.analysis.les_smagorinsky import filter_box_2d

        f = np.zeros((5, 5))
        f[2, 2] = 1.0
        ff = filter_box_2d(f, width=3)
        # peak smeared into 3x3 = 9 cells, avg = 1/9
        assert ff[2, 2] < f[2, 2]
        assert ff[2, 2] > 0
        assert np.isclose(ff[2, 2], 1.0 / 9.0)
