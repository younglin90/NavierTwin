"""Round 429 — LSTM forecaster."""

from __future__ import annotations

import pytest


class TestLSTM:
    def test_forward(self) -> None:
        torch = pytest.importorskip("torch")
        from naviertwin.core.system_id.lstm_forecast import LSTMForecaster

        m = LSTMForecaster(input_dim=2, hidden=16, n_layers=1)
        x = torch.randn(3, 10, 2)
        y = m(x)
        assert y.shape == (3, 2)

    def test_rollout(self) -> None:
        torch = pytest.importorskip("torch")
        from naviertwin.core.system_id.lstm_forecast import LSTMForecaster

        m = LSTMForecaster(input_dim=1, hidden=8)
        x = torch.randn(2, 5, 1)
        seq = m.rollout(x, n_steps=4)
        assert seq.shape == (2, 4, 1)
