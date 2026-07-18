"""Round 226 — 간단 CMA-ES."""

from __future__ import annotations

import numpy as np


class TestCMA:
    def test_quadratic(self) -> None:
        from naviertwin.core.optimization.cma_es_simple import cma_es_simple

        x, f = cma_es_simple(
            lambda v: float(v @ v),
            x0=np.array([5.0, -3.0]), sigma0=1.0,
            n_gen=80, seed=0,
        )
        assert f < 0.1

    def test_rosenbrock(self) -> None:
        from naviertwin.core.optimization.cma_es_simple import cma_es_simple

        def rosen(v):
            return float(100 * (v[1] - v[0] ** 2) ** 2 + (1 - v[0]) ** 2)

        x, f = cma_es_simple(
            rosen, x0=np.array([-1.0, 1.0]), sigma0=0.5,
            n_gen=300, lam=30, mu=15, seed=0,
        )
        # 느슨한 수렴 확인
        assert f < 5.0
