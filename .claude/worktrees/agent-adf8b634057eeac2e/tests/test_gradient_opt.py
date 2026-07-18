"""Round 132 — numpy 경사하강 옵티마이저."""

from __future__ import annotations

import numpy as np
import pytest


class TestGradOpt:
    @pytest.mark.parametrize("name", ["sgd", "momentum", "adam", "adagrad"])
    def test_converges_quadratic(self, name: str) -> None:
        from naviertwin.core.optimization.gradient_opt import (
            AdaGradOpt,
            AdamOpt,
            MomentumOpt,
            SGDOpt,
            minimize,
        )

        opts = {
            "sgd": SGDOpt(lr=0.1),
            "momentum": MomentumOpt(lr=0.05, momentum=0.9),
            "adam": AdamOpt(lr=0.1),
            "adagrad": AdaGradOpt(lr=0.5),
        }
        Q = np.array([[3.0, 0.5], [0.5, 2.0]])

        def og(x):
            return 0.5 * float(x @ Q @ x), Q @ x

        x0 = np.array([5.0, -3.0])
        x_star, hist = minimize(og, x0, opts[name], n_steps=500, tol=1e-10)
        assert np.linalg.norm(x_star) < 0.1
        # 비단조 감소는 있을 수 있으나, 최종은 초기보다 확실히 작음
        assert hist[-1] < hist[0]
