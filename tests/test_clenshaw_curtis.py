"""Round 136 — Clenshaw-Curtis 적분."""

from __future__ import annotations

import numpy as np


class TestClenshawCurtis:
    def test_cos(self) -> None:
        from naviertwin.core.numerics.chebyshev import chebyshev_points
        from naviertwin.core.numerics.clenshaw_curtis import integrate_cc

        N = 20
        x = chebyshev_points(N)
        val = integrate_cc(np.cos(x), N)
        assert abs(val - 2 * np.sin(1.0)) < 1e-10

    def test_polynomial(self) -> None:
        """∫_{-1}^{1} x^4 dx = 2/5."""
        from naviertwin.core.numerics.chebyshev import chebyshev_points
        from naviertwin.core.numerics.clenshaw_curtis import integrate_cc

        N = 10
        x = chebyshev_points(N)
        val = integrate_cc(x ** 4, N)
        assert abs(val - 2 / 5) < 1e-10

    def test_interval(self) -> None:
        """∫_0^π sin(x) dx = 2."""
        from naviertwin.core.numerics.chebyshev import chebyshev_points
        from naviertwin.core.numerics.clenshaw_curtis import integrate_cc_interval

        N = 30
        a, b = 0.0, np.pi
        x = 0.5 * (a + b) + 0.5 * (b - a) * chebyshev_points(N)
        val = integrate_cc_interval(np.sin(x), N, a, b)
        assert abs(val - 2.0) < 1e-10
