"""Round 477 — MLS."""

from __future__ import annotations

import numpy as np


class TestMLS:
    def test_quadratic(self) -> None:
        from naviertwin.core.meshless.mls import mls_eval_1d

        x = np.linspace(0, 1, 21)
        y = x ** 2
        for q in [0.2, 0.5, 0.8]:
            v = mls_eval_1d(x_query=q, x=x, y=y, h=0.2, order=2)
            assert abs(v - q ** 2) < 1e-6

    def test_linear(self) -> None:
        from naviertwin.core.meshless.mls import mls_eval_1d

        x = np.linspace(0, 1, 11)
        y = 3 * x + 1
        v = mls_eval_1d(x_query=0.5, x=x, y=y, h=0.3, order=1)
        assert abs(v - 2.5) < 1e-6
