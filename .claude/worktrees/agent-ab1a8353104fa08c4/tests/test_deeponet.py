"""Round 162 — DeepONet."""

from __future__ import annotations

import pytest


class TestDeepONet:
    def test_forward_shared_y(self) -> None:
        pytest.importorskip("torch")
        import torch

        from naviertwin.core.neural.deeponet import DeepONet

        m = DeepONet(branch_input_dim=16, trunk_input_dim=2, p=8, hidden=16)
        u = torch.randn(4, 16)
        y = torch.randn(20, 2)
        out = m(u, y)
        assert out.shape == (4, 20)

    def test_forward_batched_y(self) -> None:
        pytest.importorskip("torch")
        import torch

        from naviertwin.core.neural.deeponet import DeepONet

        m = DeepONet(branch_input_dim=8, trunk_input_dim=1, p=4, hidden=8)
        u = torch.randn(3, 8)
        y = torch.randn(3, 10, 1)
        out = m(u, y)
        assert out.shape == (3, 10)

    def test_train(self) -> None:
        pytest.importorskip("torch")
        import torch

        from naviertwin.core.neural.deeponet import DeepONet

        torch.manual_seed(0)
        m = DeepONet(branch_input_dim=8, trunk_input_dim=1, p=4, hidden=16)
        u = torch.randn(10, 8)
        y = torch.linspace(-1, 1, 20).reshape(-1, 1)
        target = torch.randn(10, 20)
        opt = torch.optim.Adam(m.parameters(), lr=1e-2)
        loss0 = ((m(u, y) - target) ** 2).mean().item()
        for _ in range(80):
            opt.zero_grad()
            loss = ((m(u, y) - target) ** 2).mean()
            loss.backward()
            opt.step()
        assert loss.item() < loss0
