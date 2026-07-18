"""Negative tests — verify the L2 MMS layer detects intentional regressions.

These tests *patch* a solver in-process to introduce a known bug, run the
MMS check, and assert the convergence/agreement check FAILS — proving our
verification net catches regressions instead of always-passing.
"""

from __future__ import annotations

import numpy as np
import pytest

from naviertwin.core.verification.loglog_slope import slope_fit

pytestmark = pytest.mark.convergence


def _broken_ssp_rk3_step(u, rhs, *, dt):
    """Bug: only one stage instead of three → drops to forward Euler (1st order)."""
    return u + dt * rhs(u)


class TestNegativeBugDetected:
    def test_broken_solver_drops_order(self) -> None:
        """A broken SSP-RK3 (= forward Euler) gives slope ≈ 1, NOT 3."""
        T = 0.5
        u_exact = np.exp(-T)

        def run(n: int) -> float:
            dt = T / n
            u = np.array([1.0])
            for _ in range(n):
                u = _broken_ssp_rk3_step(u, lambda u: -u, dt=dt)
            return float(abs(u[0] - u_exact))

        ns = [40, 80, 160]
        hs = np.array([T / n for n in ns])
        errs = np.array([run(n) for n in ns])
        p = slope_fit(hs, errs)

        # The same band the positive test uses (2.5..4.0) MUST reject it.
        assert not (2.5 <= p <= 4.0), (
            f"L2 MMS net failed to catch a broken solver! observed p={p}"
        )
        # And the broken one should be ~1st order (forward Euler).
        assert 0.5 <= p <= 1.5, f"expected ~1st order, got {p}"
