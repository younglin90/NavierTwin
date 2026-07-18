"""Round 372 — r-adaptation."""

from __future__ import annotations

import numpy as np


class TestRAdapt:
    def test_clusters_to_high_weight(self) -> None:
        from naviertwin.core.amr.r_adapt import r_adapt_1d

        x = np.linspace(0, 1, 21)
        # weight high in [0.7, 1.0]
        w = np.where(x > 0.7, 10.0, 1.0)
        x2 = r_adapt_1d(x, w, n_iter=20)
        # nodes near 0.7-1.0 should be denser
        n_dense = ((x2 >= 0.7) & (x2 <= 1.0)).sum()
        n_orig = ((x >= 0.7) & (x <= 1.0)).sum()
        assert n_dense > n_orig

    def test_endpoints_fixed(self) -> None:
        from naviertwin.core.amr.r_adapt import r_adapt_1d

        x = np.linspace(0, 1, 11)
        x2 = r_adapt_1d(x, np.ones(11), n_iter=10)
        assert x2[0] == 0.0
        assert x2[-1] == 1.0
