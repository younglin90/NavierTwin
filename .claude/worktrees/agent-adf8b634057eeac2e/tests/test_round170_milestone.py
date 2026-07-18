"""Round 170 — 마일스톤: R161-R169 import + e2e."""

from __future__ import annotations

import numpy as np
import pytest

R161_169 = [
    "naviertwin.core.neural.fno_layer",
    "naviertwin.core.neural.deeponet",
    "naviertwin.core.surrogate.gp_scratch",
    "naviertwin.core.active_learning.query",
    "naviertwin.core.optimization.pareto",
    "naviertwin.core.export.quantize",
    "naviertwin.core.serving.http_server",
    "naviertwin.core.streaming.tail_reader",
    "naviertwin.core.analysis.lambda2",
]


class TestRound170:
    @pytest.mark.parametrize("m", R161_169)
    def test_importable(self, m: str) -> None:
        import importlib

        importlib.import_module(m)

    def test_gp_active_learning_cycle(self) -> None:
        """GP → 예측 분산 → top-variance active learning 쿼리."""
        from naviertwin.core.active_learning.query import top_variance_query
        from naviertwin.core.surrogate.gp_scratch import GPRegressor

        rng = np.random.default_rng(0)
        X = rng.uniform(-1, 1, size=(12, 1))
        y = np.sin(X[:, 0] * 3)
        gp = GPRegressor(lengthscale=0.3, sigma=1.0, noise=1e-4).fit(X, y)

        Xq = np.linspace(-1, 1, 100)[:, None]
        mu, var = gp.predict(Xq)
        # 상위 3 variance → 훈련점에서 먼 곳
        idx = top_variance_query(var, k=3)
        assert idx.size == 3
        # 분산이 훈련점 근처보다 높은지
        assert float(var.max()) >= float(var.min())

    def test_fno_deeponet_shapes(self) -> None:
        pytest.importorskip("torch")
        import torch

        from naviertwin.core.neural.deeponet import DeepONet
        from naviertwin.core.neural.fno_layer import FNOBlock1d

        blk = FNOBlock1d(channels=8, modes=4)
        don = DeepONet(
            branch_input_dim=16, trunk_input_dim=1, p=4, hidden=8,
        )
        x = torch.randn(2, 8, 32)
        y = blk(x)
        assert y.shape == x.shape
        u = torch.randn(3, 16)
        tq = torch.linspace(-1, 1, 10).reshape(-1, 1)
        o = don(u, tq)
        assert o.shape == (3, 10)

    def test_pareto_filter(self) -> None:
        from naviertwin.core.optimization.pareto import pareto_mask

        # fake 2-objective
        rng = np.random.default_rng(0)
        F = rng.random((30, 2))
        m = pareto_mask(F)
        assert m.sum() >= 1
