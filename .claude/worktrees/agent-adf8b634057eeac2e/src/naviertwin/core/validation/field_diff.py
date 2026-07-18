"""두 필드 간 차이 분석 — pointwise / 글로벌 통계 / hotspot 탐지.

Examples:
    >>> import numpy as np
    >>> from naviertwin.core.validation.field_diff import field_diff_stats
    >>> a = np.zeros(100); b = np.ones(100)
    >>> s = field_diff_stats(a, b)
    >>> s["rmse"]
    1.0
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def field_diff_stats(
    a: NDArray[np.float64],
    b: NDArray[np.float64],
) -> dict[str, float]:
    """a - b 의 통계치."""
    a = np.asarray(a, dtype=np.float64).ravel()
    b = np.asarray(b, dtype=np.float64).ravel()
    if a.shape != b.shape:
        raise ValueError(f"shape 불일치: {a.shape} vs {b.shape}")
    d = a - b
    l2 = float(np.linalg.norm(d))
    ref = float(np.linalg.norm(b)) + 1e-30
    return {
        "mae": float(np.mean(np.abs(d))),
        "rmse": float(np.sqrt(np.mean(d ** 2))),
        "max_abs": float(np.max(np.abs(d))),
        "bias": float(np.mean(d)),
        "l2": l2,
        "relative_l2": l2 / ref,
    }


def hotspot_indices(
    a: NDArray[np.float64],
    b: NDArray[np.float64],
    *,
    top_k: int = 10,
) -> NDArray[np.int64]:
    """|a - b| 가장 큰 지점 top_k 개 인덱스."""
    d = np.abs(np.asarray(a).ravel() - np.asarray(b).ravel())
    top_k = int(min(top_k, d.size))
    idx = np.argpartition(-d, top_k - 1)[:top_k]
    return idx[np.argsort(-d[idx])]


def band_mask(
    a: NDArray[np.float64],
    b: NDArray[np.float64],
    *,
    tol: float = 1e-3,
) -> NDArray[np.bool_]:
    """|a - b| > tol 마스크."""
    return np.abs(np.asarray(a) - np.asarray(b)) > tol


__all__ = ["field_diff_stats", "hotspot_indices", "band_mask"]
