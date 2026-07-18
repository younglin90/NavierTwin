"""이미지 필드 비교 — PSNR / SSIM / NRMSE.

Examples:
    >>> import numpy as np
    >>> from naviertwin.core.validation.image_metrics import psnr, ssim
    >>> a = np.ones((32, 32))
    >>> b = np.ones((32, 32)) * 1.01
    >>> psnr(a, b, data_range=1.0) > 30
    True
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def psnr(
    a: NDArray[np.float64], b: NDArray[np.float64], *, data_range: float = 1.0,
) -> float:
    mse = float(np.mean((np.asarray(a) - np.asarray(b)) ** 2))
    if mse == 0:
        return float("inf")
    return float(10 * np.log10(data_range ** 2 / mse))


def nrmse(
    a: NDArray[np.float64], b: NDArray[np.float64],
) -> float:
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    rmse = float(np.sqrt(np.mean((a - b) ** 2)))
    denom = max(float(a.max() - a.min()), 1e-30)
    return rmse / denom


def ssim(
    a: NDArray[np.float64], b: NDArray[np.float64],
    *, k1: float = 0.01, k2: float = 0.03, data_range: float = 1.0,
) -> float:
    """Global SSIM (patch 없이 전체 통계)."""
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    c1 = (k1 * data_range) ** 2
    c2 = (k2 * data_range) ** 2
    mu_a = a.mean()
    mu_b = b.mean()
    va = a.var()
    vb = b.var()
    cov = float(((a - mu_a) * (b - mu_b)).mean())
    num = (2 * mu_a * mu_b + c1) * (2 * cov + c2)
    den = (mu_a ** 2 + mu_b ** 2 + c1) * (va + vb + c2)
    return float(num / (den + 1e-30))


__all__ = ["psnr", "nrmse", "ssim"]
