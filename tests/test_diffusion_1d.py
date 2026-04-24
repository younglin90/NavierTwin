"""Round 287 — diffusion 1D."""

from __future__ import annotations

import pytest


class TestDiffusion:
    def test_forward_diffusion_shape(self) -> None:
        torch = pytest.importorskip("torch")
        from naviertwin.core.neural.diffusion_1d import forward_diffusion

        x0 = torch.zeros(8, 16)
        t = torch.linspace(0.1, 1.0, 8)
        x_t, eps = forward_diffusion(x0, t)
        assert x_t.shape == x0.shape
        assert eps.shape == x0.shape
        # nonzero noise
        assert x_t.std() > 0.05

    def test_score_net_forward(self) -> None:
        torch = pytest.importorskip("torch")
        from naviertwin.core.neural.diffusion_1d import ScoreNet1D

        sn = ScoreNet1D(dim=10, hidden=32)
        x = torch.randn(4, 10)
        t = torch.rand(4)
        s = sn(x, t)
        assert s.shape == x.shape

    def test_score_train_step(self) -> None:
        torch = pytest.importorskip("torch")
        from naviertwin.core.neural.diffusion_1d import (
            ScoreNet1D,
            forward_diffusion,
        )

        sn = ScoreNet1D(dim=4, hidden=16)
        opt = torch.optim.Adam(sn.parameters(), lr=1e-2)
        x0 = torch.randn(32, 4)
        t = torch.rand(32) * 0.9 + 0.1
        x_t, eps = forward_diffusion(x0, t)
        sigma = torch.sqrt(t).unsqueeze(-1)
        target = -eps / sigma
        s = sn(x_t, t)
        loss = ((s - target) ** 2).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
        assert torch.isfinite(loss)
