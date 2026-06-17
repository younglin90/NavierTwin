"""Haar 웨이블릿 변환 — 1D 신호 다중 해상도 분해/재구성.

Examples:
    >>> import numpy as np
    >>> from naviertwin.core.analysis.wavelet import haar_forward, haar_inverse
    >>> x = np.array([1., 2., 3., 4., 5., 6., 7., 8.])
    >>> coeffs = haar_forward(x, level=2)
    >>> y = haar_inverse(coeffs)
    >>> np.allclose(x, y)
    True
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from naviertwin._native import _kernels


def _haar_step(x: NDArray[np.float64]) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    x = np.asarray(x, dtype=np.float64)
    if x.size % 2 != 0:
        raise ValueError("길이가 2의 배수이어야 함")
    inv_sqrt2 = 1.0 / np.sqrt(2.0)
    a = (x[::2] + x[1::2]) * inv_sqrt2
    d = (x[::2] - x[1::2]) * inv_sqrt2
    return a, d


def haar_forward(
    x: NDArray[np.float64], level: int = 1,
) -> dict:
    """다단계 Haar. 반환: {"approx": final, "details": [d_L, ..., d_1]}."""
    if _kernels is None:
        raise ImportError("NavierTwin native kernels are required by haar_forward")
    return _kernels.haar_forward(np.asarray(x, dtype=np.float64), level)


def haar_inverse(coeffs: dict) -> NDArray[np.float64]:
    if _kernels is None:
        raise ImportError("NavierTwin native kernels are required by haar_inverse")
    return _kernels.haar_inverse(coeffs)


def haar_threshold(
    coeffs: dict, tau: float,
) -> dict:
    """soft-thresholding 노이즈 제거."""
    if _kernels is None:
        raise ImportError("NavierTwin native kernels are required by haar_threshold")
    return _kernels.haar_threshold(coeffs, tau)


__all__ = ["haar_forward", "haar_inverse", "haar_threshold"]
