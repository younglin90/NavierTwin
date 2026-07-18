"""Round 292 — k-ω SST."""

from __future__ import annotations

import numpy as np


class TestSST:
    def test_eddy_viscosity_low_strain(self) -> None:
        from naviertwin.core.analysis.komega_sst import A1, eddy_viscosity_sst

        k = np.array([1.0])
        omega = np.array([1.0])
        S = np.array([0.0])  # no strain
        F2 = np.array([1.0])
        nu_t = eddy_viscosity_sst(k, omega, S, F2)
        # denom = max(a1*1, 0) = a1; ν_t = a1 * 1 / a1 = 1.0
        assert np.isclose(nu_t[0], 1.0)
        # high strain → limiter active
        S_high = np.array([100.0])
        nu_t_h = eddy_viscosity_sst(k, omega, S_high, F2)
        # denom ≈ S F2; ν_t = a1 k / (S F2)
        assert np.isclose(nu_t_h[0], A1 * 1.0 / (100.0 * 1.0))

    def test_F1_in_range(self) -> None:
        from naviertwin.core.analysis.komega_sst import sst_blending_F1

        k = np.array([1.0, 0.1])
        omega = np.array([1.0, 1.0])
        y = np.array([0.01, 1.0])
        F1 = sst_blending_F1(k, omega, y, nu=1e-5)
        assert ((0 <= F1) & (F1 <= 1)).all()
