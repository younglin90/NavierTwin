"""Round 289 — Conditional FNO 1D."""

from __future__ import annotations

import pytest


class TestCFNO:
    def test_forward_shape(self) -> None:
        torch = pytest.importorskip("torch")
        from naviertwin.core.neural.cfno import CFNO1D

        m = CFNO1D(in_ch=2, out_ch=2, modes=8, width=16, p_dim=3, n_layers=2)
        x = torch.randn(2, 2, 64)
        p = torch.randn(2, 3)
        y = m(x, p)
        assert y.shape == (2, 2, 64)

    def test_param_sensitivity(self) -> None:
        torch = pytest.importorskip("torch")
        from naviertwin.core.neural.cfno import CFNO1D

        torch.manual_seed(0)
        m = CFNO1D(in_ch=1, out_ch=1, modes=4, width=8, p_dim=2, n_layers=2)
        x = torch.randn(1, 1, 32)
        y1 = m(x, torch.zeros(1, 2))
        y2 = m(x, torch.ones(1, 2) * 5.0)
        assert not torch.allclose(y1, y2)
