"""PI-Latent-NO — 물리 제약 잠재 연산자.

L-DeepONet 구조에 물리 잔차 loss 를 추가하여 **라벨 데이터 없이도** 학습 가능.
사용자가 ``latent_residual_fn(z_in, z_out, dec_fn)`` 을 제공하면 그 평균 제곱을
loss 에 합산한다.

Examples:
    >>> import numpy as np
    >>> import torch
    >>> from naviertwin.core.operator_learning.latent_operator.pi_latent_no import (
    ...     PILatentNO,
    ... )
    >>> rng = np.random.default_rng(0)
    >>> X = rng.standard_normal((20, 30)).astype(np.float32)
    >>> Y = X ** 2  # 약한 supervision
    >>> def residual_fn(z_in, z_out, dec):
    ...     # 잠재 변화 크기를 제한 (예시)
    ...     return z_out - z_in
    >>> op = PILatentNO(n_features=30, latent=4, hidden=32, max_epochs=3,
    ...                 residual_fn=residual_fn, physics_weight=0.1)
    >>> op.fit({"inputs": X, "outputs": Y})
"""

from __future__ import annotations

from typing import Any, Callable

import numpy as np

from naviertwin.core.operator_learning.latent_operator.l_deeponet import LDeepONet
from naviertwin.utils.logger import get_logger

logger = get_logger(__name__)


class PILatentNO(LDeepONet):
    """L-DeepONet + 잠재 공간 residual loss."""

    def __init__(
        self,
        n_features: int,
        residual_fn: Callable[..., Any] | None = None,
        physics_weight: float = 1.0,
        **kwargs: Any,
    ) -> None:
        super().__init__(n_features=n_features, **kwargs)
        self.residual_fn = residual_fn
        self.physics_weight = physics_weight
        self.phys_losses_: list[float] = []
        self.data_losses_: list[float] = []

    def fit(self, dataset: dict[str, Any]) -> None:
        import torch
        from torch.utils.data import DataLoader, TensorDataset

        X = np.asarray(dataset["inputs"], dtype=np.float32)
        Y = np.asarray(dataset["outputs"], dtype=np.float32)
        if X.shape[1] != self.n_features or Y.shape[1] != self.n_features:
            raise ValueError("inputs/outputs 의 피처 수가 n_features 와 일치")

        self._device = self._resolve_device()
        self._build()
        tuple(map(lambda m: m.to(self._device), (self._enc, self._dec, self._op)))

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
        self.data_losses_ = []
        self.phys_losses_ = []

        epoch_idx = 0
        while epoch_idx < self.max_epochs:
            epoch_data = 0.0
            epoch_phys = 0.0
            batches = iter(loader)
            while True:
                try:
                    xb, yb = next(batches)
                except StopIteration:
                    break
                xb = xb.to(self._device)
                yb = yb.to(self._device)
                optim.zero_grad()
                z_in = self._enc(xb)
                x_rec = self._dec(z_in)
                z_out = self._op(z_in)
                y_pred = self._dec(z_out)
                data = mse(y_pred, yb) + 0.1 * mse(x_rec, xb)

                if self.residual_fn is not None:
                    res = self.residual_fn(z_in, z_out, self._dec)
                    phys = (res ** 2).mean()
                else:
                    phys = torch.tensor(0.0, device=self._device)

                loss = data + self.physics_weight * phys
                loss.backward()
                optim.step()
                epoch_data += float(data.item()) * xb.shape[0]
                epoch_phys += float(phys.item()) * xb.shape[0]

            n = max(len(X), 1)
            self.data_losses_.append(epoch_data / n)
            self.phys_losses_.append(epoch_phys / n)
            self.train_losses_.append(
                self.data_losses_[-1] + self.physics_weight * self.phys_losses_[-1]
            )
            epoch_idx += 1

        self.is_fitted = True
        logger.info(
            "PI-Latent-NO 학습 완료: data=%.6g, phys=%.6g, λ=%.2g",
            self.data_losses_[-1],
            self.phys_losses_[-1],
            self.physics_weight,
        )


__all__ = ["PILatentNO"]
