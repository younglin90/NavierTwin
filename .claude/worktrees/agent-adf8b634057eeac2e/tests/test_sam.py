"""Round 492 — SAM."""

from __future__ import annotations

import pytest


class TestSAM:
    def test_two_step(self) -> None:
        torch = pytest.importorskip("torch")
        from naviertwin.utils.sam import SAM

        m = torch.nn.Linear(4, 1)
        opt = torch.optim.SGD(m.parameters(), lr=0.01)
        sam = SAM(m.parameters(), opt, rho=0.05)
        x = torch.randn(2, 4)
        y = m(x).sum()
        y.backward()
        # capture initial param
        p0 = m.weight.detach().clone()
        sam.first_step()
        # perturbed
        assert not torch.allclose(p0, m.weight.detach())
        # restore + step
        sam.second_step()
        # should be back near p0 minus opt step
        assert torch.isfinite(m.weight).all()
