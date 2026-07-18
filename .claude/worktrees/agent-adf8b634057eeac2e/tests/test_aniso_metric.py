"""Round 308 — anisotropic metric."""

from __future__ import annotations

import numpy as np


class TestAniso:
    def test_metric_spd(self) -> None:
        from naviertwin.core.tools.aniso_metric import metric_from_hessian

        rng = np.random.default_rng(0)
        H = rng.standard_normal((3, 2, 2))
        M = metric_from_hessian(H, h_min=0.01, h_max=1.0)
        for Mi in M:
            ev = np.linalg.eigvalsh(Mi)
            assert (ev > 0).all()

    def test_edge_length(self) -> None:
        from naviertwin.core.tools.aniso_metric import edge_length_metric

        M = np.eye(2)
        L = edge_length_metric(M, M, np.array([0., 0]), np.array([1., 0]))
        assert np.isclose(L, 1.0)
        # stretched metric
        Ms = np.diag([4.0, 1.0])
        L = edge_length_metric(Ms, Ms, np.array([0., 0]), np.array([1., 0]))
        assert np.isclose(L, 2.0)
