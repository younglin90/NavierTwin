"""v5.1.0 multi-fidelity / transfer / active learning 테스트."""

from __future__ import annotations

import numpy as np
import pytest


class TestCoKriging:
    def test_basic_accuracy(self) -> None:
        pytest.importorskip("sklearn")
        from naviertwin.core.multi_fidelity.multi_fidelity import AdditiveCoKriging

        rng = np.random.default_rng(0)
        X_L = rng.uniform(-1, 1, (40, 1))
        X_H = X_L[:12]
        y_L = np.sin(2 * X_L[:, 0])
        y_H = y_L[:12] + 0.05 * np.cos(5 * X_H[:, 0])

        mf = AdditiveCoKriging()
        mf.fit(X_L, y_L, X_H, y_H)

        X_test = np.linspace(-1, 1, 20).reshape(-1, 1)
        y_pred = mf.predict(X_test)
        assert y_pred.shape == (20,)
        assert np.all(np.isfinite(y_pred))


class TestTransferLearning:
    def test_freeze_layers(self) -> None:
        pytest.importorskip("torch")
        import torch.nn as nn

        from naviertwin.core.multi_fidelity.transfer_learning import freeze_layers

        model = nn.Sequential(nn.Linear(4, 8), nn.ReLU(), nn.Linear(8, 8), nn.ReLU(), nn.Linear(8, 2))
        freeze_layers(model, n_freeze=2)
        reqs = [p.requires_grad for p in model.parameters()]
        # 처음 2 Linear 는 freeze → 처음 4 params (weight+bias × 2)
        assert reqs[:4] == [False, False, False, False]
        assert all(reqs[4:])

    def test_finetune_reduces_loss(self) -> None:
        pytest.importorskip("torch")
        import torch.nn as nn

        from naviertwin.core.multi_fidelity.transfer_learning import finetune

        model = nn.Sequential(nn.Linear(2, 16), nn.Tanh(), nn.Linear(16, 1))
        rng = np.random.default_rng(0)
        X = rng.standard_normal((30, 2)).astype(np.float32)
        Y = np.sin(X.sum(axis=1, keepdims=True)).astype(np.float32)
        losses = finetune(model, X, Y, lr=1e-2, max_epochs=20, batch_size=8, device="cpu")
        assert losses[-1] < losses[0]


class TestActiveLearning:
    def test_variance_strategy(self) -> None:
        pytest.importorskip("sklearn")
        from sklearn.gaussian_process import GaussianProcessRegressor

        from naviertwin.core.online_learning.active_learning import select_next_samples

        rng = np.random.default_rng(0)
        X = rng.standard_normal((20, 2))
        y = np.sin(X.sum(axis=1))
        gp = GaussianProcessRegressor().fit(X, y)
        pool = rng.standard_normal((60, 2))
        idx = select_next_samples(gp, pool, k=5, strategy="variance")
        assert idx.shape == (5,)
        # 모두 unique
        assert len(set(idx.tolist())) == 5

    def test_random_strategy(self) -> None:
        from naviertwin.core.online_learning.active_learning import select_next_samples

        pool = np.random.default_rng(0).standard_normal((30, 2))
        idx = select_next_samples(None, pool, k=4, strategy="random", seed=1)
        assert idx.shape == (4,)

    def test_active_loop(self) -> None:
        pytest.importorskip("sklearn")
        from sklearn.gaussian_process import GaussianProcessRegressor

        from naviertwin.core.online_learning.active_learning import active_loop

        rng = np.random.default_rng(0)
        X_init = rng.standard_normal((5, 1))
        y_init = np.sin(X_init[:, 0])
        pool = rng.standard_normal((40, 1))

        def oracle(x: np.ndarray) -> float:
            return float(np.sin(x[0]))

        X_f, y_f = active_loop(
            model_factory=GaussianProcessRegressor,
            X_init=X_init, y_init=y_init,
            pool=pool, oracle=oracle, n_query=4,
        )
        assert len(X_f) == 9
        assert len(y_f) == 9
