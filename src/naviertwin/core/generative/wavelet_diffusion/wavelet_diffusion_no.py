"""Wavelet Diffusion NO — 웨이블릿 도메인 확산 + 역변환.

1D 시그널에 대해 DWT 로 downscale, 저해상도 coefficient 에서 DDPM,
마지막에 IDWT 로 복원. PyWavelets optional.

Examples:
    >>> import numpy as np
    >>> from naviertwin.core.generative.wavelet_diffusion.wavelet_diffusion_no import (
    ...     WaveletDiffusionNO,
    ... )
    >>> rng = np.random.default_rng(0)
    >>> X = rng.standard_normal((30, 64)).astype(np.float32)
    >>> m = WaveletDiffusionNO(n_features=64, wavelet="db2", level=1, n_steps=8, max_epochs=2)
    >>> m.fit(X)
    >>> samples = m.sample(4, seed=0)
    >>> samples.shape
    (4, 64)
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from naviertwin.core.generative.diffusion_pde.diffusion_pde import DiffusionPDE
from naviertwin.utils.logger import get_logger

logger = get_logger(__name__)


def _require_pywt() -> object:
    try:
        import pywt
    except ImportError as exc:
        raise RuntimeError("pywt(PyWavelets) 필요") from exc
    return pywt


class WaveletDiffusionNO:
    """DWT → DiffusionPDE(저해상도) → IDWT 복원."""

    def __init__(
        self,
        n_features: int,
        wavelet: str = "db2",
        level: int = 1,
        n_steps: int = 100,
        max_epochs: int = 50,
        **kwargs: object,
    ) -> None:
        self.n_features = n_features
        self.wavelet = wavelet
        self.level = level
        self.n_steps = n_steps
        self.max_epochs = max_epochs
        self.kwargs = kwargs

        self._coef_shapes: list[int] = []
        self._inner: DiffusionPDE | None = None
        self.is_fitted: bool = False

    def _pack(self, X: NDArray[np.float64]) -> NDArray[np.float64]:
        """각 샘플에 대해 DWT → flat 1D 벡터 packed."""
        pywt = _require_pywt()
        out = []
        i = 0
        while i < X.shape[0]:
            coeffs = pywt.wavedec(X[i], self.wavelet, level=self.level)
            if i == 0:
                self._coef_shapes = list(map(len, coeffs))
            out.append(np.concatenate(coeffs))
            i += 1
        return np.stack(out).astype(np.float32)

    def _unpack(self, V: NDArray[np.float64]) -> NDArray[np.float64]:
        """flat → coeffs 리스트 → IDWT 로 원 시그널."""
        pywt = _require_pywt()
        out = []
        sample_idx = 0
        while sample_idx < V.shape[0]:
            v = V[sample_idx]
            coeffs: list[NDArray[np.float64]] = []
            idx = 0
            shape_idx = 0
            while shape_idx < len(self._coef_shapes):
                size = self._coef_shapes[shape_idx]
                coeffs.append(v[idx : idx + size])
                idx += size
                shape_idx += 1
            x = pywt.waverec(coeffs, self.wavelet)
            # 길이 정규화
            if x.shape[0] > self.n_features:
                x = x[: self.n_features]
            elif x.shape[0] < self.n_features:
                x = np.pad(x, (0, self.n_features - x.shape[0]))
            out.append(x)
            sample_idx += 1
        return np.stack(out).astype(np.float64)

    def fit(self, X: NDArray[np.float64]) -> None:
        X = np.asarray(X, dtype=np.float32)
        if X.ndim != 2 or X.shape[1] != self.n_features:
            raise ValueError(
                f"X shape={X.shape} — (n, {self.n_features}) 2D 필요"
            )
        V = self._pack(X)
        packed_dim = V.shape[1]
        self._inner = DiffusionPDE(
            n_features=packed_dim,
            n_steps=self.n_steps,
            max_epochs=self.max_epochs,
            **self.kwargs,
        )
        self._inner.fit(V)
        self.is_fitted = True
        logger.info(
            "WaveletDiffusionNO 학습 완료: packed=%d, level=%d",
            packed_dim, self.level,
        )

    def sample(self, n_samples: int, seed: int | None = None) -> NDArray[np.float64]:
        if not self.is_fitted or self._inner is None:
            raise RuntimeError("fit() 먼저 호출")
        V = self._inner.sample(n_samples, seed=seed)
        return self._unpack(V)


__all__ = ["WaveletDiffusionNO"]
