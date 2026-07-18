"""Round 510 — Y category milestone: V&V (R501-R509) + auto-report e2e."""

from __future__ import annotations

import numpy as np


class TestMilestoneY:
    def test_imports(self) -> None:
        from naviertwin.core.verification import (  # noqa: F401
            gci,
            loglog_slope,
            mms,
            monotone,
            order_table,
            richardson,
            sobol,
            uq_disc,
            vv20,
        )

    def test_mms_gci_e2e(self) -> None:
        """Errs at 3 grids → fit slope (≈ 2) → GCI."""
        from naviertwin.core.verification.gci import gci, observed_order
        from naviertwin.core.verification.loglog_slope import slope_fit

        h = np.array([0.1, 0.05, 0.025])
        err = h ** 2  # 2nd-order
        p = slope_fit(h, err)
        assert abs(p - 2.0) < 1e-9
        # observed order from f-values
        f = 1 + h ** 2
        p2 = observed_order(f1=f[2], f2=f[1], f3=f[0], r=2.0)
        assert abs(p2 - 2.0) < 1e-9
        eps = (f[2] - f[1]) / max(abs(f[2]), 1e-30)
        g = gci(eps=eps, r=2.0, p=2.0)
        assert g >= 0
