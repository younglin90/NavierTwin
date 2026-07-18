"""Round 213 — Neural SDE."""

from __future__ import annotations

import pytest


class TestNSDE:
    def test_shapes(self) -> None:
        pytest.importorskip("torch")
        import torch

        from naviertwin.core.neural.neural_sde import NeuralSDE

        torch.manual_seed(0)
        sde = NeuralSDE(state_dim=2, hidden=16, diffusion=0.1)
        x0 = torch.zeros(8, 2)
        traj = sde.simulate(x0, t0=0.0, t1=1.0, dt=0.1)
        assert traj.shape == (11, 8, 2)

    def test_train_drift(self) -> None:
        pytest.importorskip("torch")
        import torch

        from naviertwin.core.neural.neural_sde import NeuralSDE

        torch.manual_seed(0)
        sde = NeuralSDE(state_dim=1, hidden=8, diffusion=0.01)
        # target drift = -x
        x = torch.randn(100, 1)
        tb = torch.zeros_like(x)
        inp = torch.cat([x, tb], dim=-1)
        target = -x
        opt = torch.optim.Adam(sde.parameters(), lr=1e-2)
        l0 = ((sde.drift(inp) - target) ** 2).mean().item()
        for _ in range(200):
            opt.zero_grad()
            loss = ((sde.drift(inp) - target) ** 2).mean()
            loss.backward()
            opt.step()
        assert loss.item() < l0 * 0.5
