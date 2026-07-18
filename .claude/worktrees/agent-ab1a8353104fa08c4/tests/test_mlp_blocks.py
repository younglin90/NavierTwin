"""Round 114 — MLP / Residual / SIREN blocks."""

from __future__ import annotations

import pytest


class TestMLP:
    def test_mlp_forward(self) -> None:
        pytest.importorskip("torch")
        import torch

        from naviertwin.core.neural.mlp_blocks import MLP

        m = MLP(3, 2, hidden=[16, 16], activation="gelu")
        x = torch.randn(5, 3)
        y = m(x)
        assert y.shape == (5, 2)

    def test_mlp_train_step(self) -> None:
        pytest.importorskip("torch")
        import torch

        from naviertwin.core.neural.mlp_blocks import MLP

        m = MLP(2, 1, hidden=[8])
        x = torch.randn(20, 2)
        y = torch.randn(20, 1)
        opt = torch.optim.Adam(m.parameters(), lr=1e-3)
        loss0 = ((m(x) - y) ** 2).mean().item()
        for _ in range(100):
            opt.zero_grad()
            loss = ((m(x) - y) ** 2).mean()
            loss.backward()
            opt.step()
        assert loss.item() < loss0

    def test_residual_forward(self) -> None:
        pytest.importorskip("torch")
        import torch

        from naviertwin.core.neural.mlp_blocks import ResidualMLP

        m = ResidualMLP(4, 3, hidden=16, n_blocks=2)
        y = m(torch.randn(7, 4))
        assert y.shape == (7, 3)

    def test_siren_forward(self) -> None:
        pytest.importorskip("torch")
        import torch

        from naviertwin.core.neural.mlp_blocks import SirenMLP

        m = SirenMLP(2, 1, hidden=32, n_layers=4)
        y = m(torch.linspace(-1, 1, 20).unsqueeze(-1).expand(-1, 2))
        assert y.shape == (20, 1)

    def test_invalid_activation(self) -> None:
        pytest.importorskip("torch")
        from naviertwin.core.neural.mlp_blocks import MLP

        with pytest.raises(ValueError):
            MLP(3, 2, activation="bogus")
