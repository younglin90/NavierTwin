"""Round 291 — RANS k-ε."""

from __future__ import annotations

import numpy as np


class TestRANSKE:
    def test_eddy_viscosity_formula(self) -> None:
        from naviertwin.core.analysis.rans_ke import C_MU, eddy_viscosity_ke

        k = np.array([1.0, 4.0])
        eps = np.array([0.1, 1.0])
        nu_t = eddy_viscosity_ke(k, eps)
        assert np.allclose(nu_t, C_MU * k * k / eps)

    def test_zero_k(self) -> None:
        from naviertwin.core.analysis.rans_ke import eddy_viscosity_ke

        nu_t = eddy_viscosity_ke(np.zeros(3), np.array([0.1, 0.2, 0.3]))
        assert np.allclose(nu_t, 0.0)

    def test_production_positive(self) -> None:
        from naviertwin.core.analysis.rans_ke import production_term

        nu_t = np.array([0.1, 0.2])
        S = np.array([[[1.0, 0.5], [0.5, 1.0]], [[2.0, 0.0], [0.0, 1.0]]])
        P = production_term(nu_t, S)
        assert (P >= 0).all()
