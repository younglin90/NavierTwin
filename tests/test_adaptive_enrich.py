"""Round 407 — adaptive enrichment."""

from __future__ import annotations

import numpy as np


class TestEnrich:
    def test_orthonormal(self) -> None:
        from naviertwin.core.dimensionality_reduction.nonlinear.adaptive_enrich import (
            enrich_basis,
        )

        Phi = np.eye(5)[:, :2]
        r = np.array([0., 0, 1., 1., 0])
        Phi2 = enrich_basis(Phi, r)
        assert Phi2.shape == (5, 3)
        assert np.allclose(Phi2.T @ Phi2, np.eye(3), atol=1e-10)

    def test_no_growth_in_span(self) -> None:
        from naviertwin.core.dimensionality_reduction.linear.lspg import lspg_solve  # noqa: F401
        from naviertwin.core.dimensionality_reduction.nonlinear.adaptive_enrich import (
            enrich_basis,
        )

        Phi = np.eye(5)[:, :2]
        r = np.array([1., 2., 0, 0, 0])  # already in span
        Phi2 = enrich_basis(Phi, r)
        assert Phi2.shape == Phi.shape
