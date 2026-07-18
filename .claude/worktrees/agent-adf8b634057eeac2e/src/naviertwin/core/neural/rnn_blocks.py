"""RNN 블록 — LSTM / GRU 시퀀스 모델.

Examples:
    >>> import torch  # doctest: +SKIP
"""

from __future__ import annotations


def _torch():
    try:
        import torch

        return torch
    except ImportError as exc:
        raise RuntimeError("torch 필요") from exc


def LSTMSeq(
    input_dim: int, hidden: int = 32, output_dim: int | None = None,
    n_layers: int = 1, dropout: float = 0.0, bidirectional: bool = False,
):
    _torch()
    import torch.nn as nn

    out_d = output_dim if output_dim is not None else input_dim
    n_dir = 2 if bidirectional else 1

    class _Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.lstm = nn.LSTM(
                input_dim, hidden, num_layers=n_layers,
                dropout=dropout if n_layers > 1 else 0.0,
                bidirectional=bidirectional, batch_first=True,
            )
            self.head = nn.Linear(hidden * n_dir, out_d)

        def forward(self, x):
            y, _ = self.lstm(x)
            return self.head(y)

    return _Net()


def GRUSeq(
    input_dim: int, hidden: int = 32, output_dim: int | None = None,
    n_layers: int = 1, dropout: float = 0.0,
):
    _torch()
    import torch.nn as nn

    out_d = output_dim if output_dim is not None else input_dim

    class _Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.gru = nn.GRU(
                input_dim, hidden, num_layers=n_layers,
                dropout=dropout if n_layers > 1 else 0.0, batch_first=True,
            )
            self.head = nn.Linear(hidden, out_d)

        def forward(self, x):
            y, _ = self.gru(x)
            return self.head(y)

    return _Net()


__all__ = ["LSTMSeq", "GRUSeq"]
