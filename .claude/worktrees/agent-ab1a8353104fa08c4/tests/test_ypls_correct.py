"""Round 294 — y+ wall functions."""

from __future__ import annotations

import numpy as np


class TestYPlus:
    def test_linear_sublayer(self) -> None:
        from naviertwin.core.analysis.ypls_correct import u_plus_blended

        y = np.array([1.0, 5.0, 11.0])
        u = u_plus_blended(y)
        assert np.allclose(u, y)

    def test_log_layer(self) -> None:
        from naviertwin.core.analysis.ypls_correct import (
            KAPPA,
            B,
            u_plus_blended,
        )

        y = np.array([100.0, 1000.0])
        u = u_plus_blended(y)
        expected = (1.0 / KAPPA) * np.log(y) + B
        assert np.allclose(u, expected)

    def test_monotonic(self) -> None:
        from naviertwin.core.analysis.ypls_correct import u_plus_blended

        y = np.linspace(0.1, 1000.0, 50)
        u = u_plus_blended(y)
        assert np.all(np.diff(u) >= 0)
