"""Round 324 — SVGD."""

from __future__ import annotations

import numpy as np


class TestSVGD:
    def test_target_normal(self) -> None:
        from naviertwin.core.uncertainty.svgd import svgd_step

        rng = np.random.default_rng(0)
        x = rng.normal(5.0, 0.5, size=(30, 1))  # offset particles

        def grad_logp(x):
            return -x  # standard normal

        for _ in range(200):
            x = svgd_step(x, grad_logp, lr=0.1)
        # particles drift toward 0
        assert abs(x.mean()) < 0.5
        assert x.std() < 1.5

    def test_step_shape(self) -> None:
        from naviertwin.core.uncertainty.svgd import svgd_step

        x = np.random.default_rng(0).standard_normal((10, 2))
        x_new = svgd_step(x, lambda x: -x, lr=0.01)
        assert x_new.shape == x.shape
