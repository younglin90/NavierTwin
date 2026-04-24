"""Round 288 — RealNVP-lite."""

from __future__ import annotations

import pytest


class TestRealNVP:
    def test_forward_shape(self) -> None:
        torch = pytest.importorskip("torch")
        from naviertwin.core.neural.realnvp_lite import RealNVPLite

        flow = RealNVPLite(dim=4, hidden=16, n_layers=3)
        x = torch.randn(8, 4)
        z, logdet = flow(x)
        assert z.shape == x.shape
        assert logdet.shape == (8,)

    def test_invertibility(self) -> None:
        torch = pytest.importorskip("torch")
        from naviertwin.core.neural.realnvp_lite import RealNVPLite

        torch.manual_seed(0)
        flow = RealNVPLite(dim=6, hidden=16, n_layers=4)
        x = torch.randn(4, 6)
        z, _ = flow(x)
        x_rec = flow.inverse(z)
        assert torch.allclose(x, x_rec, atol=1e-4)
