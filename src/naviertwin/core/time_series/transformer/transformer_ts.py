"""Temporal Transformer — 어텐션 기반 장거리 시계열 예측.

Encoder-only transformer + causal mask + positional embedding.
LSTMForecaster 와 동일 인터페이스 (lookback → next step + rollout).

Examples:
    >>> import numpy as np
    >>> from naviertwin.core.time_series.transformer.transformer_ts import TransformerForecaster
    >>> rng = np.random.default_rng(0)
    >>> seqs = rng.standard_normal((4, 40, 3)).astype(np.float32)
    >>> m = TransformerForecaster(n_features=3, d_model=16, n_heads=2, lookback=8, max_epochs=3)
    >>> m.fit({"sequences": seqs})
    >>> m.predict(seqs[0, :8], n_steps=5).shape
    (5, 3)
"""

from __future__ import annotations

from typing import Any

import numpy as np

from naviertwin.core.time_series.base import BaseTimeSeries
from naviertwin.core.time_series.lstm.lstm import _build_sliding_windows
from naviertwin.utils.logger import get_logger

logger = get_logger(__name__)


class TransformerForecaster(BaseTimeSeries):
    """Causal TransformerEncoder 기반 시계열 예측."""

    def __init__(
        self,
        n_features: int,
        d_model: int = 32,
        n_heads: int = 2,
        n_layers: int = 2,
        dim_ff: int = 64,
        lookback: int = 8,
        max_epochs: int = 50,
        batch_size: int = 32,
        lr: float = 1e-3,
        dropout: float = 0.0,
        device: str = "auto",
        seed: int | None = 0,
    ) -> None:
        super().__init__(device=device)
        self.n_features = n_features
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.dim_ff = dim_ff
        self.lookback = lookback
        self.max_epochs = max_epochs
        self.batch_size = batch_size
        self.lr = lr
        self.dropout = dropout
        self.seed = seed

        self._model: Any = None
        self._device: Any = None
        self.train_losses_: list[float] = []

    def _resolve_device(self) -> Any:
        import torch

        if self.device == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return torch.device(self.device)

    def _build(self) -> Any:
        import torch
        import torch.nn as nn

        if self.seed is not None:
            torch.manual_seed(self.seed)

        d = self.d_model
        n_feat = self.n_features
        L = self.lookback

        class _TF(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.proj_in = nn.Linear(n_feat, d)
                self.pos = nn.Parameter(torch.zeros(1, L, d))
                enc_layer = nn.TransformerEncoderLayer(
                    d_model=d,
                    nhead=self.n_heads_,
                    dim_feedforward=self.dim_ff_,
                    dropout=self.dropout_,
                    batch_first=True,
                )
                self.enc = nn.TransformerEncoder(enc_layer, num_layers=self.n_layers_)
                self.head = nn.Linear(d, n_feat)

            def forward(self, x: Any) -> Any:  # (B, L, F)
                Lc = x.shape[1]
                mask = torch.triu(
                    torch.ones(Lc, Lc, device=x.device), diagonal=1
                ).bool()
                x = self.proj_in(x) + self.pos[:, :Lc, :]
                h = self.enc(x, mask=mask)
                return self.head(h[:, -1, :])

        _TF.n_heads_ = self.n_heads
        _TF.dim_ff_ = self.dim_ff
        _TF.dropout_ = self.dropout
        _TF.n_layers_ = self.n_layers
        return _TF()

    def fit(self, dataset: dict[str, Any]) -> None:
        import torch
        from torch.utils.data import DataLoader, TensorDataset

        seqs = np.asarray(dataset["sequences"], dtype=np.float32)
        if seqs.ndim != 3:
            raise ValueError(f"sequences (N,T,F) 3D 필요: {seqs.shape}")
        if seqs.shape[1] <= self.lookback:
            raise ValueError(f"T({seqs.shape[1]}) <= lookback({self.lookback})")

        X, Y = _build_sliding_windows(seqs, self.lookback)
        self._device = self._resolve_device()
        self._model = self._build().to(self._device)
        optim = torch.optim.Adam(self._model.parameters(), lr=self.lr)
        loss_fn = torch.nn.MSELoss()

        loader = DataLoader(
            TensorDataset(torch.tensor(X), torch.tensor(Y)),
            batch_size=min(self.batch_size, len(X)),
            shuffle=True,
        )
        self.train_losses_ = []
        for _ in range(self.max_epochs):
            epoch_loss = 0.0
            for xb, yb in loader:
                xb = xb.to(self._device)
                yb = yb.to(self._device)
                optim.zero_grad()
                pred = self._model(xb)
                loss = loss_fn(pred, yb)
                loss.backward()
                optim.step()
                epoch_loss += float(loss.item()) * xb.shape[0]
            epoch_loss /= max(len(X), 1)
            self.train_losses_.append(epoch_loss)

        self.is_fitted = True
        logger.info(
            "TransformerForecaster 학습 완료: d=%d heads=%d loss=%.6g",
            self.d_model,
            self.n_heads,
            self.train_losses_[-1] if self.train_losses_ else float("nan"),
        )

    def predict(self, initial_state: np.ndarray, n_steps: int) -> np.ndarray:
        import torch

        self._check_fitted()
        if n_steps < 1:
            raise ValueError(f"n_steps 는 1 이상: {n_steps}")
        state = np.asarray(initial_state, dtype=np.float32)
        if state.ndim == 1:
            state = np.tile(state[None, :], (self.lookback, 1))
        if state.shape[0] < self.lookback:
            pad = np.repeat(state[:1], self.lookback - state.shape[0], axis=0)
            state = np.concatenate([pad, state], axis=0)
        window = state[-self.lookback :].copy()

        preds: list[np.ndarray] = []
        for _ in range(n_steps):
            x = torch.tensor(window[None, :, :], device=self._device)
            with torch.no_grad():
                yhat = self._model(x).cpu().numpy()[0]
            preds.append(yhat)
            window = np.concatenate([window[1:], yhat[None, :]], axis=0)
        return np.stack(preds)


__all__ = ["TransformerForecaster"]
