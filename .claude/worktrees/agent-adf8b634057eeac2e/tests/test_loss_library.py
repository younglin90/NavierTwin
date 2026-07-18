"""Round 253 — loss library."""

from __future__ import annotations

import numpy as np
import pytest


class TestLoss:
    def test_mse_mae(self) -> None:
        from naviertwin.core.neural.loss_library import mae, mse, rmse

        y = np.array([1.0, 2.0, 3.0])
        yh = np.array([1.5, 2.5, 2.5])
        assert mse(y, yh) == pytest.approx(0.25)
        assert mae(y, yh) == pytest.approx(0.5)
        assert abs(rmse(y, yh) - 0.5) < 1e-10

    def test_huber(self) -> None:
        from naviertwin.core.neural.loss_library import huber

        # small residual → quadratic
        assert huber(np.array([0.0]), np.array([0.1]), delta=1.0) == pytest.approx(0.005)
        # large residual → linear
        assert huber(np.array([0.0]), np.array([5.0]), delta=1.0) == pytest.approx(4.5)

    def test_relative_l2(self) -> None:
        from naviertwin.core.neural.loss_library import relative_l2

        y = np.array([3.0, 4.0])
        yh = np.array([3.0, 4.0])
        assert relative_l2(y, yh) == pytest.approx(0.0, abs=1e-12)

    def test_quantile(self) -> None:
        from naviertwin.core.neural.loss_library import quantile_loss

        # median q=0.5 → 0.5 * |d|
        ql = quantile_loss(np.array([0.0]), np.array([4.0]), q=0.5)
        assert ql == pytest.approx(2.0)

    def test_logcosh_smape(self) -> None:
        from naviertwin.core.neural.loss_library import logcosh, smape

        assert logcosh(np.zeros(3), np.zeros(3)) == pytest.approx(0.0, abs=1e-12)
        assert 0 <= smape(np.array([1.0]), np.array([1.1])) < 0.2
