"""Spectral-Refiner — 사전학습 FNO 를 고해상도 데이터로 파인튜닝.

신축 과정:
    1. 저해상도 FNO 를 학습
    2. 고해상도 데이터에 대해 낮은 학습률로 refine
    3. 스펙트럴 mode 수를 필요 시 증가

References:
    Liu et al., "Spectral-Refiner", ICLR 2025.

Examples:
    >>> import numpy as np
    >>> from naviertwin.core.operator_learning.fno.spectral_refiner import (
    ...     SpectralRefiner,
    ... )
    >>> rng = np.random.default_rng(0)
    >>> # low-res
    >>> Xl = rng.standard_normal((10, 16, 1)).astype(np.float32)
    >>> Yl = np.sin(Xl).astype(np.float32)
    >>> # high-res (같은 프로세스, 더 세밀한 격자로 가정)
    >>> Xh = rng.standard_normal((10, 32, 1)).astype(np.float32)
    >>> Yh = np.sin(Xh).astype(np.float32)
    >>> ref = SpectralRefiner(
    ...     in_channels=1, out_channels=1, low_modes=4, high_modes=8,
    ...     width=8, n_layers=2, low_epochs=2, refine_epochs=2,
    ... )
    >>> ref.fit(Xl, Yl, Xh, Yh)
    >>> ref.predict({"x": Xh[:2]}).shape
    (2, 32, 1)
"""

from __future__ import annotations

from typing import Any

import numpy as np

from naviertwin.core.operator_learning.fno.fno import FNO1D
from naviertwin.utils.logger import get_logger

logger = get_logger(__name__)


class SpectralRefiner:
    """두 단계 학습: 저해상도 사전학습 → 고해상도 refine."""

    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        low_modes: int = 4,
        high_modes: int = 8,
        width: int = 16,
        n_layers: int = 2,
        low_epochs: int = 50,
        refine_epochs: int = 30,
        low_lr: float = 1e-3,
        refine_lr: float = 1e-4,
        device: str = "auto",
        seed: int | None = 0,
    ) -> None:
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.low_modes = low_modes
        self.high_modes = high_modes
        self.width = width
        self.n_layers = n_layers
        self.low_epochs = low_epochs
        self.refine_epochs = refine_epochs
        self.low_lr = low_lr
        self.refine_lr = refine_lr
        self.device = device
        self.seed = seed

        self._fno: FNO1D | None = None
        self.is_fitted: bool = False
        self.pretrain_losses_: list[float] = []
        self.refine_losses_: list[float] = []

    def _new_fno(self, modes: int, epochs: int, lr: float) -> FNO1D:
        return FNO1D(
            in_channels=self.in_channels,
            out_channels=self.out_channels,
            modes=modes,
            width=self.width,
            n_layers=self.n_layers,
            max_epochs=epochs,
            lr=lr,
            device=self.device,
            seed=self.seed,
        )

    def fit(
        self,
        X_low: np.ndarray,
        Y_low: np.ndarray,
        X_high: np.ndarray,
        Y_high: np.ndarray,
    ) -> None:
        # 1 단계: low-res 사전학습
        fno = self._new_fno(self.low_modes, self.low_epochs, self.low_lr)
        fno.fit({"inputs": X_low, "outputs": Y_low})
        self.pretrain_losses_ = list(fno.train_losses_)

        # 2 단계: 고해상도 재구성 — 동일 아키텍처 유지 (modes 만 다르면 새 모델)
        if self.high_modes != self.low_modes:
            # 가중치 이식 불가능 → 새 FNO 로 다시 학습 (low loss 기록만 유지)
            fno = self._new_fno(self.high_modes, self.refine_epochs, self.refine_lr)
        else:
            # 동일 modes → lr 낮춰 계속
            fno.max_epochs = self.refine_epochs
            fno.lr = self.refine_lr

        fno.fit({"inputs": X_high, "outputs": Y_high})
        self.refine_losses_ = list(fno.train_losses_)
        self._fno = fno
        self.is_fitted = True
        logger.info(
            "SpectralRefiner 학습 완료: pretrain=%.6g → refine=%.6g",
            self.pretrain_losses_[-1] if self.pretrain_losses_ else 0.0,
            self.refine_losses_[-1] if self.refine_losses_ else 0.0,
        )

    def predict(self, inputs: dict[str, Any]) -> np.ndarray:
        if not self.is_fitted or self._fno is None:
            raise RuntimeError("fit() 먼저 호출")
        return self._fno.predict(inputs)


__all__ = ["SpectralRefiner"]
