"""Round 557 — chiller COP."""

from __future__ import annotations

import numpy as np


class TestChiller:
    def test_corner(self) -> None:
        from naviertwin.core.applied.chiller_cop import interpolate_cop

        T_cw = np.array([20.0, 30.0])
        T_chw = np.array([5.0, 10.0])
        table = np.array([[5.0, 4.5], [4.0, 3.5]])
        # at corner (20, 5) → 5.0
        assert interpolate_cop(T_cw_q=20.0, T_chw_q=5.0, T_cw=T_cw,
                                  T_chw=T_chw, COP=table) == 5.0

    def test_center(self) -> None:
        from naviertwin.core.applied.chiller_cop import interpolate_cop

        T_cw = np.array([20.0, 30.0])
        T_chw = np.array([5.0, 10.0])
        table = np.array([[5.0, 4.5], [4.0, 3.5]])
        c = interpolate_cop(T_cw_q=25.0, T_chw_q=7.5, T_cw=T_cw,
                              T_chw=T_chw, COP=table)
        # average of 4 corners: (5+4.5+4+3.5)/4 = 4.25
        assert abs(c - 4.25) < 1e-12
