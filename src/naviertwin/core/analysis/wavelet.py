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

from math import sqrt

import numpy as np
from numpy.typing import NDArray


def _haar_step(x: NDArray[np.float64]) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    x = np.asarray(x, dtype=np.float64)
    if x.size % 2 != 0:
        raise ValueError("길이가 2의 배수이어야 함")
    a = (x[::2] + x[1::2]) / sqrt(2.0)
    d = (x[::2] - x[1::2]) / sqrt(2.0)
    return a, d


def haar_forward(
    x: NDArray[np.float64], level: int = 1,
) -> dict:
    """다단계 Haar. 반환: {"approx": final, "details": [d_L, ..., d_1]}."""
    a = np.asarray(x, dtype=np.float64).copy()
    details: list[NDArray[np.float64]] = []
    for _ in range(level):
        a, d = _haar_step(a)
        details.append(d)
    return {"approx": a, "details": details[::-1]}  # fine → coarse 순


def haar_inverse(coeffs: dict) -> NDArray[np.float64]:
    a = coeffs["approx"].copy()
    for d in coeffs["details"][::-1]:  # coarse → fine
        up = np.zeros(2 * a.size)
        up[::2] = (a + d) / sqrt(2.0)
        up[1::2] = (a - d) / sqrt(2.0)
        a = up
    return a


def haar_threshold(
    coeffs: dict, tau: float,
) -> dict:
    """soft-thresholding 노이즈 제거."""
    out = {"approx": coeffs["approx"], "details": []}
    for d in coeffs["details"]:
        out["details"].append(np.sign(d) * np.maximum(np.abs(d) - tau, 0.0))
    return out


__all__ = ["haar_forward", "haar_inverse", "haar_threshold"]
