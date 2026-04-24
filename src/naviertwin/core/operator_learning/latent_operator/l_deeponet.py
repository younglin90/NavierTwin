"""L-DeepONet — AE 잠재공간 + DeepONet.

고차원 필드를 AE 로 잠재 공간으로 압축한 뒤, 그 위에서 DeepONet 을 학습.
예측 시 잠재 결과를 디코딩하여 원 해상도 필드 복원.

dataset 규약 (fit):
    - ``"inputs"``: (n_samples, n_features) — 입력 함수 (고차원).
    - ``"outputs"``: (n_samples, n_features) — 출력 함수 (같은 공간).
    - ``"trunk_inputs"``: 선택 — 없으면 1D 좌표 자동 생성 사용.

Examples:
    >>> import numpy as np
    >>> from naviertwin.core.operator_learning.latent_operator.l_deeponet import (
    ...     LDeepONet,
    ... )
    >>> rng = np.random.default_rng(0)
    >>> X = rng.standard_normal((30, 50)).astype(np.float32)
    >>> Y = np.tanh(X).astype(np.float32)
    >>> op = LDeepONet(n_features=50, latent=6, hidden=32, max_epochs=3)
    >>> op.fit({"inputs": X, "outputs": Y})
    >>> op.predict({"x": X[:2]}).shape
    (2, 50)
"""

from __future__ import annotations

from typing import Any

import numpy as np

from naviertwin.core.operator_learning.base import BaseOperator
from naviertwin.utils.logger import get_logger

logger = get_logger(__name__)


class LDeepONet(BaseOperator):
    """AE 인코더 공유 + 잠재 DeepONet.

    구조:
        z_in  = enc(x)
        z_out = branch(z_in) ⊙ trunk(y)        (trunk 생략 시 단순 MLP head)
        y_hat = dec(z_out_tiled)
    """

    def __init__(
        self,
        n_features: int,
        latent: int = 16,
        hidden: int = 64,
        n_enc_layers: int = 3,
        n_dec_layers: int = 3,
        max_epochs: int = 100,
        batch_size: int = 32,
        lr: float = 1e-3,
        device: str = "auto",
        seed: int | None = 0,
    ) -> None:
        super().__init__(device=device)
        self.n_features = n_features
        self.latent = latent
        self.hidden = hidden
        self.n_enc_layers = n_enc_layers
        self.n_dec_layers = n_dec_layers
        self.max_epochs = max_epochs
        self.batch_size = batch_size
        self.lr = lr
        self.seed = seed

        self._enc: Any = None
        self._dec: Any = None
        self._op: Any = None  # latent→latent MLP (branch role)
        self._device: Any = None
        self.train_losses_: list[float] = []

    def _resolve_device(self) -> Any:
        import torch

        if self.device == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return torch.device(self.device)

    def _mlp(self, dims: list[int]) -> Any:
        import torch.nn as nn

        layers: list[Any] = []
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            if i < len(dims) - 2:
                layers.append(nn.GELU())
        return nn.Sequential(*layers)

    def _build(self) -> None:
        import torch

        if self.seed is not None:
            torch.manual_seed(self.seed)
        F, Z, H = self.n_features, self.latent, self.hidden
        self._enc = self._mlp([F] + [H] * (self.n_enc_layers - 1) + [Z])
        self._dec = self._mlp([Z] + [H] * (self.n_dec_layers - 1) + [F])
        self._op = self._mlp([Z, H, H, Z])

    def fit(self, dataset: dict[str, Any]) -> None:
        import torch
        from torch.utils.data import DataLoader, TensorDataset

        X = np.asarray(dataset["inputs"], dtype=np.float32)
        Y = np.asarray(dataset["outputs"], dtype=np.float32)
        if X.shape[1] != self.n_features or Y.shape[1] != self.n_features:
            raise ValueError("inputs/outputs 의 피처 수가 n_features 와 일치해야 합니다")

        self._device = self._resolve_device()
        self._build()
        for m in (self._enc, self._dec, self._op):
            m.to(self._device)

        params = (
            list(self._enc.parameters())
            + list(self._dec.parameters())
            + list(self._op.parameters())
        )
        optim = torch.optim.Adam(params, lr=self.lr)
        mse = torch.nn.MSELoss()

        loader = DataLoader(
            TensorDataset(torch.tensor(X), torch.tensor(Y)),
            batch_size=min(self.batch_size, len(X)),
            shuffle=True,
        )
        self.train_losses_ = []
        for _ in range(self.max_epochs):
            epoch = 0.0
            for xb, yb in loader:
                xb = xb.to(self._device)
                yb = yb.to(self._device)
                optim.zero_grad()
                z_in = self._enc(xb)
                x_rec = self._dec(z_in)
                z_out = self._op(z_in)
                y_pred = self._dec(z_out)
                loss = mse(y_pred, yb) + 0.1 * mse(x_rec, xb)
                loss.backward()
                optim.step()
                epoch += float(loss.item()) * xb.shape[0]
            epoch /= max(len(X), 1)
            self.train_losses_.append(epoch)

        self.is_fitted = True
        logger.info("L-DeepONet 학습 완료: loss=%.6g", self.train_losses_[-1])

    def predict(self, inputs: dict[str, Any]) -> np.ndarray:
        import torch

        self._check_fitted()
        x = np.asarray(inputs["x"], dtype=np.float32)
        squeeze = x.ndim == 1
        if squeeze:
            x = x[None, :]
        with torch.no_grad():
            xt = torch.tensor(x, device=self._device)
            z_in = self._enc(xt)
            z_out = self._op(z_in)
            y = self._dec(z_out).cpu().numpy()
        return y[0] if squeeze else y


__all__ = ["LDeepONet"]
