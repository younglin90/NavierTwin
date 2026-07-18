"""Adaptive FNO — 데이터 기반 스펙트럴 모드 수 선택.

1D: 데이터의 rFFT 평균 파워 스펙트럼에서 상위 frac 비율의 모드만 유지.

Examples:
    >>> import numpy as np
    >>> from naviertwin.core.operator_learning.fno.adaptive_fno import AdaptiveFNO1D
    >>> rng = np.random.default_rng(0)
    >>> X = rng.standard_normal((20, 32, 1)).astype(np.float32)
    >>> Y = np.sin(X).astype(np.float32)
    >>> op = AdaptiveFNO1D(
    ...     in_channels=1, out_channels=1, width=8,
    ...     energy_threshold=0.9, n_layers=2, max_epochs=2,
    ... )
    >>> op.fit({"inputs": X, "outputs": Y})
    >>> op.modes_selected_ > 0
    True
"""

from __future__ import annotations

from typing import Any

import numpy as np

from naviertwin.core.operator_learning.fno.fno import FNO1D
from naviertwin.utils.logger import get_logger

logger = get_logger(__name__)


class AdaptiveFNO1D:
    """스펙트럼 에너지 기반 modes 자동 선택 wrapper."""

    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        width: int = 32,
        energy_threshold: float = 0.95,
        min_modes: int = 2,
        max_modes: int | None = None,
        n_layers: int = 4,
        max_epochs: int = 100,
        batch_size: int = 16,
        lr: float = 1e-3,
        device: str = "auto",
        seed: int | None = 0,
    ) -> None:
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.width = width
        self.energy_threshold = energy_threshold
        self.min_modes = min_modes
        self.max_modes = max_modes
        self.n_layers = n_layers
        self.max_epochs = max_epochs
        self.batch_size = batch_size
        self.lr = lr
        self.device = device
        self.seed = seed

        self.modes_selected_: int = 0
        self._fno: FNO1D | None = None
        self.is_fitted: bool = False

    def _select_modes(self, Y: np.ndarray) -> int:
        """Y: (B, N, C) 의 rFFT 평균 파워에서 cumulative energy 가 임계 넘는 모드 수."""
        power = np.abs(np.fft.rfft(Y, axis=1)).mean(axis=(0, 2))
        energy = power ** 2
        cum = np.cumsum(energy) / max(energy.sum(), 1e-30)
        k = int(np.searchsorted(cum, self.energy_threshold) + 1)
        k = max(k, self.min_modes)
        if self.max_modes is not None:
            k = min(k, self.max_modes)
        k = min(k, energy.size)
        return k

    def fit(self, dataset: dict[str, Any]) -> None:
        Y = np.asarray(dataset["outputs"], dtype=np.float32)
        if Y.ndim != 3:
            raise ValueError(f"outputs (B,N,C) 3D 필요: {Y.shape}")
        self.modes_selected_ = self._select_modes(Y)
        logger.info(
            "AdaptiveFNO1D: 선택 modes=%d (energy≥%.3f)",
            self.modes_selected_, self.energy_threshold,
        )
        self._fno = FNO1D(
            in_channels=self.in_channels,
            out_channels=self.out_channels,
            modes=self.modes_selected_,
            width=self.width,
            n_layers=self.n_layers,
            max_epochs=self.max_epochs,
            batch_size=self.batch_size,
            lr=self.lr,
            device=self.device,
            seed=self.seed,
        )
        self._fno.fit(dataset)
        self.is_fitted = True

    def predict(self, inputs: dict[str, Any]) -> np.ndarray:
        if not self.is_fitted or self._fno is None:
            raise RuntimeError("fit() 먼저 호출")
        return self._fno.predict(inputs)


__all__ = ["AdaptiveFNO1D"]
