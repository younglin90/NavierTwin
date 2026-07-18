"""Round 505 — order-of-accuracy table."""

from __future__ import annotations

import numpy as np


class TestOrderTable:
    def test_consistent_p(self) -> None:
        from naviertwin.core.verification.order_table import order_table

        h = np.array([0.2, 0.1, 0.05, 0.025])
        err = h ** 2
        t = order_table(h, err)
        for p in t["p_pair"]:
            assert abs(p - 2.0) < 1e-9
