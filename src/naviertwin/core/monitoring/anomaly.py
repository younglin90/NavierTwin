"""잔차 기반 이상 탐지 — threshold / CUSUM / 지수가중이동평균.

Examples:
    >>> import numpy as np
    >>> from naviertwin.core.monitoring.anomaly import threshold_detector
    >>> x = np.array([0.1, 0.2, 0.3, 5.0, 0.2, -4.0])
    >>> threshold_detector(x, threshold=2.0).tolist()
    [False, False, False, True, False, True]
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def threshold_detector(
    residuals: NDArray[np.float64], threshold: float,
) -> NDArray[np.bool_]:
    return np.abs(np.asarray(residuals)) > threshold


def zscore_detector(
    residuals: NDArray[np.float64], k: float = 3.0,
) -> NDArray[np.bool_]:
    r = np.asarray(residuals, dtype=np.float64)
    mu = r.mean()
    sd = r.std() + 1e-30
    return np.abs(r - mu) / sd > k


def cusum(
    residuals: NDArray[np.float64], *,
    k: float = 0.5, h: float = 5.0,
) -> NDArray[np.bool_]:
    """Positive/negative CUSUM: 드리프트 누적 감지."""
    r = np.asarray(residuals, dtype=np.float64)
    Splus = 0.0
    Sminus = 0.0
    alarms = np.zeros(r.size, dtype=bool)
    for i, v in enumerate(r):
        Splus = max(0.0, Splus + v - k)
        Sminus = min(0.0, Sminus + v + k)
        alarms[i] = Splus > h or Sminus < -h
    return alarms


def ewma(
    residuals: NDArray[np.float64], *, lam: float = 0.2, k: float = 3.0,
) -> NDArray[np.bool_]:
    """EWMA control chart."""
    r = np.asarray(residuals, dtype=np.float64)
    mu = r[0]
    z = np.zeros_like(r)
    z[0] = mu
    for i in range(1, r.size):
        z[i] = (1 - lam) * z[i - 1] + lam * r[i]
    sigma = r.std() + 1e-30
    # stdized
    return np.abs(z - mu) / sigma > k


__all__ = ["threshold_detector", "zscore_detector", "cusum", "ewma"]
