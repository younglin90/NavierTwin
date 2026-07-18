"""Round 493 — Lipschitz."""

from __future__ import annotations

import numpy as np


class TestLip:
    def test_linear(self) -> None:
        from naviertwin.utils.lipschitz import lipschitz_sampled

        L = lipschitz_sampled(lambda x: 3.0 * x, np.linspace(-1, 1, 100))
        # exactly 3
        assert abs(L - 3.0) < 1e-9

    def test_constant(self) -> None:
        from naviertwin.utils.lipschitz import lipschitz_sampled

        L = lipschitz_sampled(lambda x: 5.0 * np.ones_like(x), np.linspace(-1, 1, 50))
        assert L == 0.0
