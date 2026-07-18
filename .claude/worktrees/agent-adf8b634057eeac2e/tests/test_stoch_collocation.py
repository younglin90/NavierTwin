"""Round 328 — sparse grid collocation."""

from __future__ import annotations

import numpy as np


class TestSparse:
    def test_cc_quadrature_constant(self) -> None:
        from naviertwin.core.uncertainty.stoch_collocation import (
            clenshaw_curtis_nodes,
        )

        x, w = clenshaw_curtis_nodes(level=4)
        # ∫_{-1}^{1} 1 dx = 2
        assert np.isclose(w.sum(), 2.0, atol=1e-8)

    def test_cc_quadrature_x2(self) -> None:
        from naviertwin.core.uncertainty.stoch_collocation import (
            clenshaw_curtis_nodes,
        )

        x, w = clenshaw_curtis_nodes(level=5)
        # ∫_{-1}^{1} x² dx = 2/3
        assert np.isclose((w * x * x).sum(), 2.0 / 3.0, atol=1e-8)

    def test_sparse_grid_runs(self) -> None:
        from naviertwin.core.uncertainty.stoch_collocation import sparse_grid_2d

        pts, w = sparse_grid_2d(level=3)
        assert pts.shape[1] == 2
        assert pts.shape[0] == w.shape[0]
