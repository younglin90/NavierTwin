"""C4 회전 대칭 equivariant FNO2D (간이 구현).

표준 FNO2D forward 를 4 개 회전 그룹 원소 {0°, 90°, 180°, 270°} 에
대해 실행하고 출력을 역회전 후 평균내어 equivariance 를 근사적으로
확보한다. e3nn 없이 PyTorch 만으로 동작.

A rigorous G-equivariance implementation should parameterize the spectral
weights directly in an equivariant basis; this wrapper is a pragmatic
augmentation-averaging approximation suitable in CFD cases where
exact equivariance is not strictly required but the bias helps.

Examples:
    >>> import numpy as np
    >>> from naviertwin.core.equivariant.group_equiv_fno.group_equiv_fno import (
    ...     C4EquivariantFNO2D,
    ... )
    >>> rng = np.random.default_rng(0)
    >>> X = rng.standard_normal((4, 16, 16, 1)).astype(np.float32)
    >>> Y = X ** 2
    >>> op = C4EquivariantFNO2D(in_channels=1, out_channels=1, modes1=4, modes2=4,
    ...                         width=8, n_layers=2, max_epochs=2)
    >>> op.fit({"inputs": X, "outputs": Y})
    >>> op.predict({"x": X[:2]}).shape
    (2, 16, 16, 1)
"""

from __future__ import annotations

from typing import Any

import numpy as np

from naviertwin.core.operator_learning.base import BaseOperator
from naviertwin.core.operator_learning.fno.fno import FNO2D
from naviertwin.utils.logger import get_logger

logger = get_logger(__name__)


class C4EquivariantFNO2D(BaseOperator):
    """FNO2D + C4 group averaging wrapper."""

    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        modes1: int = 8,
        modes2: int = 8,
        width: int = 16,
        n_layers: int = 4,
        max_epochs: int = 100,
        batch_size: int = 8,
        lr: float = 1e-3,
        device: str = "auto",
        seed: int | None = 0,
    ) -> None:
        super().__init__(device=device)
        self._fno = FNO2D(
            in_channels=in_channels,
            out_channels=out_channels,
            modes1=modes1,
            modes2=modes2,
            width=width,
            n_layers=n_layers,
            max_epochs=max_epochs,
            batch_size=batch_size,
            lr=lr,
            device=device,
            seed=seed,
        )

    def _augment_dataset(
        self, X: np.ndarray, Y: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """C4 회전 증강 — axis (1, 2) 90°씩 4 번 회전."""
        ks = range(4)
        Xs = tuple(map(lambda k: np.ascontiguousarray(np.rot90(X, k=k, axes=(1, 2))), ks))
        Ys = tuple(map(lambda k: np.ascontiguousarray(np.rot90(Y, k=k, axes=(1, 2))), range(4)))
        return np.concatenate(Xs, axis=0), np.concatenate(Ys, axis=0)

    def fit(self, dataset: dict[str, Any]) -> None:
        X = np.asarray(dataset["inputs"], dtype=np.float32)
        Y = np.asarray(dataset["outputs"], dtype=np.float32)
        if X.ndim != 4 or Y.ndim != 4:
            raise ValueError(f"(B,H,W,C) 4D 필요: {X.shape}, {Y.shape}")
        X_aug, Y_aug = self._augment_dataset(X, Y)
        self._fno.fit({"inputs": X_aug, "outputs": Y_aug})
        self.is_fitted = True
        logger.info(
            "C4EquivariantFNO2D 학습 완료: 4× 증강 (총 %d 샘플)", X_aug.shape[0]
        )

    def predict(self, inputs: dict[str, Any]) -> np.ndarray:
        """회전 평균 예측: 각 회전된 입력을 예측 후 역회전 → 평균."""
        self._check_fitted()
        x = np.asarray(inputs["x"], dtype=np.float32)
        squeeze = x.ndim == 3
        if squeeze:
            x = x[None, ...]

        ks = np.arange(4)
        rotated = np.concatenate(
            tuple(map(lambda k: np.ascontiguousarray(np.rot90(x, k=int(k), axes=(1, 2))), ks)),
            axis=0,
        )
        rotated_preds = self._fno.predict({"x": rotated})
        chunks = np.split(rotated_preds, 4, axis=0)
        preds = tuple(
            map(
                lambda item: np.ascontiguousarray(
                    np.rot90(item[1], k=-int(item[0]), axes=(1, 2))
                ),
                zip(ks, chunks, strict=True),
            )
        )
        mean = np.mean(np.stack(preds), axis=0)
        return mean[0] if squeeze else mean

    @property
    def train_losses_(self) -> list[float]:
        return self._fno.train_losses_


__all__ = ["C4EquivariantFNO2D"]
