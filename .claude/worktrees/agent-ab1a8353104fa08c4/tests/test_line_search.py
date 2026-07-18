"""Round 133 — line search."""

from __future__ import annotations

import numpy as np


class TestLineSearch:
    def test_armijo_reduces(self) -> None:
        from naviertwin.core.optimization.line_search import armijo_backtrack

        def f(x): return float(x @ x)
        def g(x): return 2 * x

        x = np.array([3.0, -2.0])
        p = -g(x)
        alpha = armijo_backtrack(f, g, x, p, alpha0=1.0)
        assert alpha > 0
        assert f(x + alpha * p) < f(x)

    def test_wolfe_conditions(self) -> None:
        from naviertwin.core.optimization.line_search import (
            armijo_backtrack,
            check_wolfe,
        )

        def f(x): return float(x @ x)
        def g(x): return 2 * x

        x = np.array([5.0, 4.0])
        p = -g(x)
        alpha = armijo_backtrack(f, g, x, p)
        c = check_wolfe(f, g, x, p, alpha)
        assert c["armijo"] is True
