"""Round 272 — DEIM."""

from __future__ import annotations

import numpy as np


class TestDEIM:
    def test_index_count_and_unique(self) -> None:
        from naviertwin.core.dimensionality_reduction.linear.deim import deim

        rng = np.random.default_rng(0)
        Q, _ = np.linalg.qr(rng.standard_normal((20, 5)))
        P, idx = deim(Q)
        assert P.shape == (20, 5)
        assert idx.shape == (5,)
        assert len(set(idx.tolist())) == 5  # all unique

    def test_interpolation_recovers_basis_vector(self) -> None:
        """f exactly in span(U) → DEIM 재구성 == f."""
        from naviertwin.core.dimensionality_reduction.linear.deim import (
            deim,
            deim_project,
        )

        rng = np.random.default_rng(1)
        U, _ = np.linalg.qr(rng.standard_normal((30, 4)))
        P, idx = deim(U)
        f = U @ np.array([1.0, -2.0, 0.5, 3.0])
        f_rec = deim_project(U, P, f[idx])
        assert np.allclose(f_rec, f, atol=1e-8)
