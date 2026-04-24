"""Round 269 — Chebyshev collocation 2D."""

from __future__ import annotations

import numpy as np


class TestCheb:
    def test_diff_matrix_shape(self) -> None:
        from naviertwin.core.solvers.cheb_2d import cheb_diff_matrix

        D, x = cheb_diff_matrix(8)
        assert D.shape == (9, 9)
        assert x.shape == (9,)
        # Cheb extrema endpoints ±1
        assert np.isclose(x[0], 1.0)
        assert np.isclose(x[-1], -1.0)

    def test_diff_polynomial(self) -> None:
        """d/dx of x² should be 2x, exact on Chebyshev."""
        from naviertwin.core.solvers.cheb_2d import cheb_diff_matrix

        D, x = cheb_diff_matrix(16)
        u = x ** 2
        du = D @ u
        assert np.allclose(du, 2 * x, atol=1e-10)

    def test_laplacian_shape(self) -> None:
        from naviertwin.core.solvers.cheb_2d import laplacian_2d

        L, x = laplacian_2d(8)
        assert L.shape == (81, 81)
        assert x.shape == (9,)
