"""Round 44 — Chebyshev spectral + Lagrange."""

from __future__ import annotations

import numpy as np
import pytest


class TestChebyshevPoints:
    def test_endpoints(self) -> None:
        from naviertwin.core.numerics.chebyshev import chebyshev_points

        x = chebyshev_points(8)
        assert x[0] == pytest.approx(1.0)
        assert x[-1] == pytest.approx(-1.0)

    def test_invalid_N(self) -> None:
        from naviertwin.core.numerics.chebyshev import chebyshev_points

        with pytest.raises(ValueError):
            chebyshev_points(0)


class TestChebyshevDiff:
    def test_constant_gives_zero(self) -> None:
        from naviertwin.core.numerics.chebyshev import (
            chebyshev_diff_matrix,
            chebyshev_points,
        )

        N = 8
        x = chebyshev_points(N)
        D = chebyshev_diff_matrix(N)
        u = np.ones_like(x)
        du = D @ u
        assert np.allclose(du, 0.0, atol=1e-10)

    def test_polynomial_exact(self) -> None:
        """x² 의 미분은 정확히 2x."""
        from naviertwin.core.numerics.chebyshev import (
            chebyshev_diff_matrix,
            chebyshev_points,
        )

        N = 10
        x = chebyshev_points(N)
        D = chebyshev_diff_matrix(N)
        u = x ** 3
        du = D @ u
        assert np.allclose(du, 3 * x ** 2, atol=1e-10)


class TestLagrange:
    def test_recovers_exact_at_nodes(self) -> None:
        from naviertwin.core.numerics.chebyshev import (
            chebyshev_points,
            lagrange_interp_1d,
        )

        xk = chebyshev_points(5)
        yk = np.sin(np.pi * xk)
        y_rec = lagrange_interp_1d(xk, yk, xk)
        assert np.allclose(y_rec, yk, atol=1e-10)

    def test_smooth_function_converges(self) -> None:
        from naviertwin.core.numerics.chebyshev import (
            chebyshev_points,
            lagrange_interp_1d,
        )

        xk = chebyshev_points(16)
        yk = np.exp(-xk ** 2)
        xt = np.linspace(-1, 1, 50)
        yt = lagrange_interp_1d(xk, yk, xt)
        yt_exact = np.exp(-xt ** 2)
        err = float(np.max(np.abs(yt - yt_exact)))
        assert err < 1e-8
