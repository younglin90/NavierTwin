"""DBSCAN 클러스터링 (scratch) — 밀도 기반 이상치/클러스터 탐지.

Examples:
    >>> import numpy as np
    >>> from naviertwin.core.analysis.dbscan import dbscan
    >>> pts = np.random.default_rng(0).standard_normal((50, 2))
    >>> labels = dbscan(pts, eps=0.5, min_samples=3)
    >>> labels.shape
    (50,)
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from naviertwin._native import _kernels

if _kernels is None:  # pragma: no cover
    raise ImportError("NavierTwin native kernels are required")


def dbscan(
    points: NDArray[np.float64], eps: float = 0.5, min_samples: int = 5,
) -> NDArray[np.int64]:
    """간단 DBSCAN. 반환: label (N,), -1 = noise."""
    return np.asarray(
        _kernels.dbscan(np.asarray(points, dtype=np.float64), float(eps), int(min_samples)),
        dtype=np.int64,
    )


def n_clusters(labels: NDArray[np.int64]) -> int:
    return int(len(set(labels.tolist()) - {-1}))


__all__ = ["dbscan", "n_clusters"]
