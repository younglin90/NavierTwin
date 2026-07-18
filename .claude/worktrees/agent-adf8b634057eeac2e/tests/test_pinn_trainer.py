"""Round 192 — PINN trainer."""

from __future__ import annotations

import pytest


class TestPINN:
    def test_fit_linear(self) -> None:
        pytest.importorskip("torch")
        import torch
        import torch.nn as nn

        from naviertwin.core.neural.pinn_trainer import PINNTrainer

        torch.manual_seed(0)
        m = nn.Sequential(nn.Linear(1, 16), nn.Tanh(), nn.Linear(16, 1))
        x = torch.linspace(-1, 1, 50).unsqueeze(-1)
        y = 2 * x + 1  # linear target

        def data_loss(model):
            return ((model(x) - y) ** 2).mean()

        # physics loss: u''(x) = 0 (linear)
        def physics_loss(model):
            xr = x.clone().requires_grad_(True)
            u = model(xr)
            du = torch.autograd.grad(u, xr, torch.ones_like(u), create_graph=True)[0]
            d2u = torch.autograd.grad(du, xr, torch.ones_like(du), create_graph=True)[0]
            return (d2u ** 2).mean()

        # boundary loss: u(-1) = -1, u(1) = 3 (2x+1)
        def bc_loss(model):
            xb = torch.tensor([[-1.0], [1.0]])
            yb = torch.tensor([[-1.0], [3.0]])
            return ((model(xb) - yb) ** 2).mean()

        trainer = PINNTrainer(
            m, lr=1e-2,
            weights={"data": 1.0, "physics": 0.1, "boundary": 1.0},
        )
        hist = trainer.train(
            {"data": data_loss, "physics": physics_loss, "boundary": bc_loss},
            n_epochs=200,
        )
        assert hist[-1]["total"] < hist[0]["total"]
        # Check final prediction close to linear
        pred = m(x).detach().squeeze()
        assert ((pred - y.squeeze()) ** 2).mean().item() < 0.1

    def test_invalid_optimizer(self) -> None:
        pytest.importorskip("torch")
        import torch.nn as nn

        from naviertwin.core.neural.pinn_trainer import PINNTrainer

        with pytest.raises(ValueError):
            PINNTrainer(nn.Linear(1, 1), optimizer="bogus")
