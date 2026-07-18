"""Round 191 — LSTM/GRU blocks."""

from __future__ import annotations

import pytest


class TestRNN:
    def test_lstm_shape(self) -> None:
        pytest.importorskip("torch")
        import torch

        from naviertwin.core.neural.rnn_blocks import LSTMSeq

        m = LSTMSeq(input_dim=4, hidden=16, output_dim=2, n_layers=2)
        y = m(torch.randn(3, 10, 4))
        assert y.shape == (3, 10, 2)

    def test_gru_shape(self) -> None:
        pytest.importorskip("torch")
        import torch

        from naviertwin.core.neural.rnn_blocks import GRUSeq

        m = GRUSeq(input_dim=3, hidden=8)
        y = m(torch.randn(2, 5, 3))
        assert y.shape == (2, 5, 3)

    def test_bidirectional(self) -> None:
        pytest.importorskip("torch")
        import torch

        from naviertwin.core.neural.rnn_blocks import LSTMSeq

        m = LSTMSeq(input_dim=2, hidden=4, output_dim=1, bidirectional=True)
        y = m(torch.randn(1, 6, 2))
        assert y.shape == (1, 6, 1)

    def test_train(self) -> None:
        pytest.importorskip("torch")
        import torch

        from naviertwin.core.neural.rnn_blocks import LSTMSeq

        torch.manual_seed(0)
        m = LSTMSeq(input_dim=2, hidden=8, output_dim=1)
        x = torch.randn(4, 10, 2)
        y = torch.randn(4, 10, 1)
        opt = torch.optim.Adam(m.parameters(), lr=1e-2)
        l0 = ((m(x) - y) ** 2).mean().item()
        for _ in range(80):
            opt.zero_grad()
            loss = ((m(x) - y) ** 2).mean()
            loss.backward()
            opt.step()
        assert loss.item() < l0
