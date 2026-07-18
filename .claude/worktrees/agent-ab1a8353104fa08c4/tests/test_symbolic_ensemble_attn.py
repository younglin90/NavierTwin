"""Round 13 — symbolic regression + ensemble/MoE + attention viz."""

from __future__ import annotations

import numpy as np
import pytest


class TestSymbolicRegression:
    def test_polynomial_fallback_recovers(self) -> None:
        from naviertwin.core.explainability.symbolic_regression import SymbolicRegressor

        rng = np.random.default_rng(0)
        X = rng.uniform(-1, 1, (200, 2))
        y = 2.0 * X[:, 0] + 0.5 * X[:, 1] ** 2
        sr = SymbolicRegressor(max_degree=2)
        sr.fit(X, y)
        assert sr.is_fitted
        y_hat = sr.predict(X)
        err = float(np.linalg.norm(y - y_hat) / np.linalg.norm(y))
        assert err < 0.2

    def test_unfitted_raises(self) -> None:
        from naviertwin.core.explainability.symbolic_regression import SymbolicRegressor

        sr = SymbolicRegressor()
        with pytest.raises(RuntimeError):
            sr.predict(np.zeros((1, 2)))


class TestEnsembleSurrogate:
    def test_mean_and_std(self) -> None:
        pytest.importorskip("smt")
        from naviertwin.core.surrogate.ensemble import EnsembleSurrogate
        from naviertwin.core.surrogate.rbf_surrogate import RBFSurrogate

        rng = np.random.default_rng(0)
        X = rng.uniform(-1, 1, (30, 2))
        y = np.sin(X[:, 0]) + X[:, 1] ** 2

        ens = EnsembleSurrogate([RBFSurrogate() for _ in range(3)])
        ens.fit(X, y.reshape(-1, 1))
        mean = ens.predict(X[:5])
        m2, std = ens.predict_with_std(X[:5])
        assert mean.shape[0] == 5
        assert std.shape[0] == 5
        assert np.all(std >= 0)


class TestMixtureOfExperts:
    def test_moe_fit_predict(self) -> None:
        pytest.importorskip("smt")
        from naviertwin.core.surrogate.ensemble import MixtureOfExperts
        from naviertwin.core.surrogate.rbf_surrogate import RBFSurrogate

        rng = np.random.default_rng(0)
        X = np.vstack([
            rng.standard_normal((20, 2)) * 0.1 + np.array([1, 1]),
            rng.standard_normal((20, 2)) * 0.1 + np.array([-1, -1]),
        ])
        y = (X[:, 0] + X[:, 1]).reshape(-1, 1)
        moe = MixtureOfExperts(
            experts=[RBFSurrogate() for _ in range(2)], n_clusters=2, seed=0,
        )
        moe.fit(X, y)
        pred = moe.predict(X[:3])
        assert pred.shape[0] == 3


class TestAttentionViz:
    def test_extract_shape(self) -> None:
        pytest.importorskip("torch")
        import torch
        import torch.nn as nn

        from naviertwin.core.explainability.attention_viz import (
            extract_attention,
            topk_attention_tokens,
        )

        mha = nn.MultiheadAttention(embed_dim=8, num_heads=2, batch_first=True)
        x = torch.randn(2, 6, 8)
        out, weights = extract_attention(mha, x)
        assert weights.shape == (2, 6, 6)
        top = topk_attention_tokens(weights, k=2)
        assert top.shape == (2, 6, 2)
