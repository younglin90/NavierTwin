"""Round 157 — PCE 간단 버전."""

from __future__ import annotations

import numpy as np
import pytest


class TestPCE:
    def test_fit_polynomial(self) -> None:
        from naviertwin.core.uncertainty.pce_simple import PCESimple

        rng = np.random.default_rng(0)
        xi = rng.uniform(-1, 1, size=(200, 1))
        y = 1.0 + 2.0 * xi[:, 0] - 0.5 * xi[:, 0] ** 2
        pce = PCESimple(order=3, family="legendre").fit(xi, y)
        xi_test = np.linspace(-1, 1, 50).reshape(-1, 1)
        y_test = 1.0 + 2.0 * xi_test[:, 0] - 0.5 * xi_test[:, 0] ** 2
        y_hat = pce.predict(xi_test)
        assert np.max(np.abs(y_hat - y_test)) < 1e-6

    def test_hermite(self) -> None:
        from naviertwin.core.uncertainty.pce_simple import PCESimple

        rng = np.random.default_rng(0)
        xi = rng.standard_normal(size=(300, 1))
        y = xi[:, 0] ** 2
        pce = PCESimple(order=4, family="hermite").fit(xi, y)
        y_hat = pce.predict(xi)
        assert np.mean((y_hat - y) ** 2) < 0.1

    def test_invalid(self) -> None:
        from naviertwin.core.uncertainty.pce_simple import PCESimple

        with pytest.raises(ValueError):
            PCESimple(family="bogus")
