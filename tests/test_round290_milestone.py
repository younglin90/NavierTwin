"""Round 290 — C category milestone: neural operators imports + forward shapes."""

from __future__ import annotations

import pytest


class TestMilestoneC:
    def test_imports(self) -> None:
        pytest.importorskip("torch")
        from naviertwin.core.neural import (  # noqa: F401
            cfno,
            diffusion_1d,
            equivariant_cnn,
            gat,
            hypernet,
            lno,
            meshgraphnet,
            op_transformer,
            realnvp_lite,
        )

    def test_gat_fno_pipeline(self) -> None:
        torch = pytest.importorskip("torch")
        from naviertwin.core.neural.cfno import CFNO1D
        from naviertwin.core.neural.gat import GATLayer

        gat = GATLayer(in_dim=4, out_dim=8, heads=1)
        x = torch.randn(6, 4)
        edges = torch.tensor([[0, 1, 2, 3, 4, 5], [1, 2, 3, 4, 5, 0]])
        h = gat(x, edges)
        assert h.shape == (6, 8)
        # downstream: pretend h is a 1D field, run through CFNO
        cfno_m = CFNO1D(in_ch=1, out_ch=1, modes=2, width=4, p_dim=1, n_layers=1)
        f = h.t().unsqueeze(0).mean(dim=1, keepdim=True)  # (1, 1, 6)
        p = torch.zeros(1, 1)
        y = cfno_m(f, p)
        assert y.shape == (1, 1, 6)
