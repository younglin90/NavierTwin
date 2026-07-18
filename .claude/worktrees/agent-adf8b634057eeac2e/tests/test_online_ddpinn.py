"""Round 21 — OnlineKriging + OnlineNN + DomainDecompPINN + PhysicsNEMOWrapper."""

from __future__ import annotations

import numpy as np
import pytest


class TestOnlineKriging:
    def test_initialize_and_update(self) -> None:
        pytest.importorskip("sklearn")
        from naviertwin.core.online_learning.online_learning import OnlineKriging

        rng = np.random.default_rng(0)
        X = rng.standard_normal((8, 2))
        y = X[:, 0] + X[:, 1]
        ok = OnlineKriging(buffer_size=20, refit_every=3)
        ok.initialize(X, y)
        for _ in range(5):
            xi = rng.standard_normal(2)
            ok.update(xi, float(xi.sum()))
        pred = ok.predict(np.array([[0.0, 0.0]]))
        assert pred.shape == (1,)

    def test_not_initialized_raises(self) -> None:
        from naviertwin.core.online_learning.online_learning import OnlineKriging

        ok = OnlineKriging()
        with pytest.raises(RuntimeError):
            ok.update(np.zeros(2), 0.0)


class TestOnlineNN:
    def test_partial_fit(self) -> None:
        pytest.importorskip("torch")
        import torch.nn as nn

        from naviertwin.core.online_learning.online_learning import OnlineNN

        model = nn.Sequential(nn.Linear(2, 8), nn.Tanh(), nn.Linear(8, 1))
        learner = OnlineNN(model, lr=1e-2, device="cpu")
        rng = np.random.default_rng(0)
        X = rng.standard_normal((6, 2)).astype(np.float32)
        y = np.sin(X.sum(axis=1, keepdims=True)).astype(np.float32)
        losses = learner.update(X, y, n_steps=5)
        assert len(losses) == 5
        # 5 스텝 학습 후 예측 shape
        pred = learner.predict(X)
        assert pred.shape == (6, 1)


class TestDDPINN:
    def test_split_and_fit(self) -> None:
        pytest.importorskip("torch")
        import torch

        from naviertwin.core.physnemo.dd_pinn import DomainDecompPINN

        def residual(model: object, x: torch.Tensor) -> torch.Tensor:
            x = x.requires_grad_(True)
            u = model(x)
            du = torch.autograd.grad(u.sum(), x, create_graph=True)[0]
            d2u = torch.autograd.grad(du.sum(), x, create_graph=True)[0]
            return d2u + (np.pi ** 2) * torch.sin(np.pi * x)

        dd = DomainDecompPINN(
            n_sub=2, domain=(0.0, 1.0),
            n_collocation=40, hidden=16, n_layers=3,
            max_epochs=200, lr=5e-3,
        )
        bc = {
            "x": np.array([[0.0], [1.0]], dtype=np.float32),
            "u": np.array([[0.0], [0.0]], dtype=np.float32),
        }
        dd.fit(residual, bc)

        # 예측 값이 유한하고 shape 이 올바르게 반환되는지 확인 (수렴은 epoch 에 의존)
        y = dd.predict(np.array([[0.2], [0.5], [0.8]]))
        assert y.shape == (3, 1)
        assert np.all(np.isfinite(y))
        assert dd.is_fitted

    def test_invalid_n_sub(self) -> None:
        from naviertwin.core.physnemo.dd_pinn import DomainDecompPINN

        with pytest.raises(ValueError):
            DomainDecompPINN(n_sub=0)


class TestPhysicsNEMOWrapper:
    def test_fallback_path(self) -> None:
        pytest.importorskip("torch")
        import torch

        from naviertwin.core.physnemo.physnemo_wrapper import PhysicsNEMOWrapper

        def residual(model: object, x: torch.Tensor) -> torch.Tensor:
            x = x.requires_grad_(True)
            u = model(x)
            du = torch.autograd.grad(u.sum(), x, create_graph=True)[0]
            d2u = torch.autograd.grad(du.sum(), x, create_graph=True)[0]
            return d2u + (np.pi ** 2) * torch.sin(np.pi * x)

        w = PhysicsNEMOWrapper(equation="poisson_1d", hidden=16, max_epochs=100)
        col = np.linspace(0, 1, 32, dtype=np.float32).reshape(-1, 1)
        bc = {
            "x": np.array([[0.0], [1.0]], dtype=np.float32),
            "u": np.array([[0.0], [0.0]], dtype=np.float32),
        }
        w.fit(residual, col, bc)
        u = w.predict(np.array([[0.5]]))
        assert u.shape == (1, 1)
