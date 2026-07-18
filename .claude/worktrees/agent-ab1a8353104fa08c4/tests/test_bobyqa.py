"""Round 335 — BOBYQA-lite."""

from __future__ import annotations

import numpy as np


class TestBOBYQA:
    def test_quadratic_2d(self) -> None:
        from naviertwin.core.optimization.bobyqa import bobyqa_lite

        x = bobyqa_lite(
            f=lambda x: float((x[0] - 2.0) ** 2 + (x[1] + 1.0) ** 2),
            x0=np.zeros(2), rho=0.5, rho_min=1e-3, max_iter=200,
        )
        assert np.allclose(x, [2.0, -1.0], atol=1e-2)

    def test_rosenbrock_progress(self) -> None:
        from naviertwin.core.optimization.bobyqa import bobyqa_lite

        def f(x):
            return float(100 * (x[1] - x[0] ** 2) ** 2 + (1 - x[0]) ** 2)

        x = bobyqa_lite(f, x0=np.array([-1.0, 1.0]), rho=0.5, rho_min=1e-4,
                        max_iter=500)
        assert f(x) < f(np.array([-1.0, 1.0]))
