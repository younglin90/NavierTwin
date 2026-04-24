"""LSTM 시계열 예측 모델.

자기회귀(autoregressive) 방식:
    lookback window 내 과거 → 다음 1 스텝 예측, 롤아웃으로 n_steps 생성.

Examples:
    >>> import numpy as np
    >>> from naviertwin.core.time_series.lstm.lstm import LSTMForecaster
    >>> rng = np.random.default_rng(0)
    >>> seqs = rng.standard_normal((5, 50, 4)).astype(np.float32)  # 5 seq × 50 step × 4 feat
    >>> model = LSTMForecaster(n_features=4, hidden=16, lookback=5, max_epochs=3)
    >>> model.fit({"sequences": seqs})
    >>> fut = model.predict(seqs[0, :5], n_steps=10)
"""

from __future__ import annotations

from typing import Any

import numpy as np

from naviertwin.core.time_series.base import BaseTimeSeries
from naviertwin.utils.logger import get_logger

logger = get_logger(__name__)


def _build_sliding_windows(
    seqs: np.ndarray, lookback: int
) -> tuple[np.ndarray, np.ndarray]:
    """(n_seq, T, F) → (N, lookback, F), (N, F) 슬라이딩 윈도우."""
    X: list[np.ndarray] = []
    Y: list[np.ndarray] = []
    for s in seqs:
        T = s.shape[0]
        for t in range(T - lookback):
            X.append(s[t : t + lookback])
            Y.append(s[t + lookback])
    return np.asarray(X, dtype=np.float32), np.asarray(Y, dtype=np.float32)


class LSTMForecaster(BaseTimeSeries):
    """단층 LSTM + 선형 헤드 자기회귀 예측기."""

    def __init__(
        self,
        n_features: int,
        hidden: int = 32,
        n_layers: int = 1,
        lookback: int = 8,
        max_epochs: int = 50,
        batch_size: int = 32,
        lr: float = 1e-3,
        device: str = "auto",
        seed: int | None = 0,
    ) -> None:
        super().__init__(device=device)
        self.n_features = n_features
        self.hidden = hidden
        self.n_layers = n_layers
        self.lookback = lookback
        self.max_epochs = max_epochs
        self.batch_size = batch_size
        self.lr = lr
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

        hidden = self.hidden
        n_feat = self.n_features
        n_layers = self.n_layers

        class _LSTM(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.rnn = nn.LSTM(n_feat, hidden, n_layers, batch_first=True)
                self.head = nn.Linear(hidden, n_feat)

            def forward(self, x: Any) -> Any:  # (B, L, F)
                out, _ = self.rnn(x)
                return self.head(out[:, -1, :])

        return _LSTM()

    def fit(self, dataset: dict[str, Any]) -> None:
        import torch
        from torch.utils.data import DataLoader, TensorDataset

        seqs = np.asarray(dataset["sequences"], dtype=np.float32)
        if seqs.ndim != 3:
            raise ValueError(f"sequences 는 (N, T, F) 3D 필요: {seqs.shape}")
        if seqs.shape[2] != self.n_features:
            raise ValueError(
                f"sequences F({seqs.shape[2]}) != n_features({self.n_features})"
            )
        if seqs.shape[1] <= self.lookback:
            raise ValueError(
                f"T({seqs.shape[1]}) <= lookback({self.lookback}) — 더 긴 시계열 필요"
            )

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
            "LSTMForecaster 학습 완료: hidden=%d lookback=%d loss=%.6g",
            self.hidden,
            self.lookback,
            self.train_losses_[-1] if self.train_losses_ else float("nan"),
        )

    def predict(self, initial_state: np.ndarray, n_steps: int) -> np.ndarray:
        import torch

        self._check_fitted()
        if n_steps < 1:
            raise ValueError(f"n_steps 는 1 이상: {n_steps}")

        state = np.asarray(initial_state, dtype=np.float32)
        if state.ndim == 1:
            # feature 만 주어진 경우 → lookback 만큼 복제
            state = np.tile(state[None, :], (self.lookback, 1))
        if state.shape[0] < self.lookback:
            # padding by first
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


__all__ = ["LSTMForecaster"]
