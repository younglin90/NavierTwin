"""Round 183 — Lyapunov exponent."""

from __future__ import annotations

import numpy as np


class TestLyap:
    def test_logistic_r4(self) -> None:
        from naviertwin.core.analysis.lyapunov import lyapunov_map

        def f(x):
            return 4 * x * (1 - x)

        def fp(x):
            return 4 - 8 * x

        lle = lyapunov_map(f, fp, x0=0.4, n=10000)
        assert abs(lle - np.log(2)) < 0.05

    def test_logistic_r3_9(self) -> None:
        from naviertwin.core.analysis.lyapunov import lyapunov_map

        r = 3.9

        def f(x):
            return r * x * (1 - x)

        def fp(x):
            return r - 2 * r * x

        lle = lyapunov_map(f, fp, x0=0.3, n=10000)
        assert lle > 0  # chaotic

    def test_stable_fixed_point(self) -> None:
        from naviertwin.core.analysis.lyapunov import lyapunov_map

        # f(x) = 0.5 x → 고정점 0 안정 → λ = log(0.5) < 0
        lle = lyapunov_map(
            lambda x: 0.5 * x, lambda x: 0.5, x0=0.1, n=500, warmup=0,
        )
        assert abs(lle - np.log(0.5)) < 1e-10
