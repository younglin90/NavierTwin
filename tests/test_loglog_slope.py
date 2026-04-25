"""Round 504 — log-log slope."""

from __future__ import annotations

import numpy as np


class TestLogLog:
    def test_second_order(self) -> None:
        from naviertwin.core.verification.loglog_slope import slope_fit

        h = np.array([0.1, 0.05, 0.025, 0.0125])
        err = h ** 2
        assert abs(slope_fit(h, err) - 2.0) < 1e-6

    def test_first_order(self) -> None:
        from naviertwin.core.verification.loglog_slope import slope_fit

        h = np.array([0.1, 0.05, 0.025])
        err = h
        assert abs(slope_fit(h, err) - 1.0) < 1e-6
