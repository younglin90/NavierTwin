"""LSTM forecaster — torch.nn.LSTM univariate forecast stub.

Examples:
    >>> import pytest
    >>> torch = pytest.importorskip("torch")
    >>> from naviertwin.core.system_id.lstm_forecast import LSTMForecaster
"""

from __future__ import annotations

import torch
from torch import nn


class LSTMForecaster(nn.Module):
    def __init__(
        self,
        input_dim: int = 1,
        hidden: int = 32,
        n_layers: int = 1,
    ) -> None:
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden, num_layers=n_layers, batch_first=True)
        self.head = nn.Linear(hidden, input_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, T, input_dim) → ŷ: (B, input_dim)."""
        out, _ = self.lstm(x)
        return self.head(out[:, -1, :])

    def rollout(self, x0: torch.Tensor, n_steps: int) -> torch.Tensor:
        """x0: (B, T, input_dim); rollout n_steps autoregressively."""
        seq = x0
        out = []
        step = 0
        while step < n_steps:
            y = self.forward(seq)
            out.append(y)
            seq = torch.cat([seq[:, 1:, :], y.unsqueeze(1)], dim=1)
            step += 1
        return torch.stack(out, dim=1)


__all__ = ["LSTMForecaster"]
