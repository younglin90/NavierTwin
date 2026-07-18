"""Round 258 — successive halving."""

from __future__ import annotations

import numpy as np


class TestHalving:
    def test_quadratic(self) -> None:
        from naviertwin.core.optimization.halving import successive_halving

        configs = [{"x": float(v)} for v in np.linspace(-2, 4, 16)]

        # 목적: maximize -(x-1)^2 — budget 따라 noise 감소
        def evaluator(cfg, budget):
            rng = np.random.default_rng(int(budget * 1000 + hash(cfg["x"]) % 1000))
            noise = rng.normal(0, 1.0 / budget)
            return -(cfg["x"] - 1) ** 2 + noise

        best, hist = successive_halving(
            configs, evaluator, max_budget=50, eta=3,
        )
        assert abs(best["x"] - 1.0) < 1.5
        assert len(hist) > 0

    def test_history_budgets_increase(self) -> None:
        from naviertwin.core.optimization.halving import successive_halving

        configs = [{"i": i} for i in range(8)]

        def evaluator(cfg, budget):
            return -abs(cfg["i"] - 3) + 0.001 * budget

        _, hist = successive_halving(configs, evaluator, max_budget=64, eta=2)
        budgets = [h["budget"] for h in hist]
        # budget 은 시간이 지남에 따라 증가
        assert budgets[0] <= budgets[-1]
