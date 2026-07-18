"""Round 218 — Neural ODE."""

from __future__ import annotations

import pytest


class TestNODE:
    def test_shape(self) -> None:
        pytest.importorskip("torch")
        import torch

        from naviertwin.core.neural.neural_ode import NeuralODE

        node = NeuralODE(state_dim=2, hidden=16)
        x0 = torch.randn(5, 2)
        traj = node(x0, 0.0, 1.0, 0.1)
        assert traj.shape == (11, 5, 2)

    def test_learn_decay(self) -> None:
        """target: dx/dt = -x → x(T) ≈ x0 e^-T."""
        pytest.importorskip("torch")
        import torch

        from naviertwin.core.neural.neural_ode import NeuralODE

        torch.manual_seed(0)
        node = NeuralODE(state_dim=1, hidden=8)
        x0 = torch.tensor([[1.0]])
        target = torch.tensor([[float(torch.exp(torch.tensor(-0.5)))]])
        opt = torch.optim.Adam(node.parameters(), lr=1e-2)
        for _ in range(200):
            opt.zero_grad()
            traj = node(x0, 0.0, 0.5, 0.05)
            loss = ((traj[-1] - target) ** 2).mean()
            loss.backward()
            opt.step()
        assert loss.item() < 0.01
