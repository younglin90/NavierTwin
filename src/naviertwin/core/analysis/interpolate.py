"""간단한 점 집합 보간 — KNN / IDW (inverse distance weighting).

Examples:
    >>> import numpy as np
    >>> from naviertwin.core.analysis.interpolate import idw_interpolate
    >>> src = np.array([[0.,0.,0.],[1.,0.,0.]])
    >>> val = np.array([0., 10.])
    >>> y = idw_interpolate(src, val, np.array([[0.5, 0., 0.]]))
    >>> abs(y[0] - 5.0) < 0.1
    True
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def _knn(
    src: NDArray[np.float64], tgt: NDArray[np.float64], k: int,
) -> tuple[NDArray[np.float64], NDArray[np.int64]]:
    # brute force (작은 데이터용). 큰 데이터는 scipy.spatial.cKDTree 사용 권장.
    d = np.linalg.norm(src[None, :, :] - tgt[:, None, :], axis=2)
    idx = np.argpartition(d, min(k - 1, d.shape[1] - 1), axis=1)[:, :k]
    # 재정렬 오름차순
    d_k = np.take_along_axis(d, idx, axis=1)
    order = np.argsort(d_k, axis=1)
    idx = np.take_along_axis(idx, order, axis=1)
    dist = np.take_along_axis(d_k, order, axis=1)
    return dist, idx


def knn_interpolate(
    src_points: NDArray[np.float64],
    values: NDArray[np.float64],
    tgt_points: NDArray[np.float64],
    k: int = 1,
) -> NDArray[np.float64]:
    """k 최근접 이웃 평균."""
    src = np.asarray(src_points, dtype=np.float64)
    tgt = np.asarray(tgt_points, dtype=np.float64)
    v = np.asarray(values, dtype=np.float64)
    _, idx = _knn(src, tgt, k)
    return v[idx].mean(axis=1)


def idw_interpolate(
    src_points: NDArray[np.float64],
    values: NDArray[np.float64],
    tgt_points: NDArray[np.float64],
    *,
    power: float = 2.0,
    k: int | None = None,
    eps: float = 1e-12,
) -> NDArray[np.float64]:
    """Inverse-distance weighting 보간. k=None → 모든 점 사용."""
    src = np.asarray(src_points, dtype=np.float64)
    tgt = np.asarray(tgt_points, dtype=np.float64)
    v = np.asarray(values, dtype=np.float64)
    if k is None or k >= src.shape[0]:
        d = np.linalg.norm(src[None, :, :] - tgt[:, None, :], axis=2)
        w = 1.0 / (d ** power + eps)
        w = w / w.sum(axis=1, keepdims=True)
        return w @ v
    d, idx = _knn(src, tgt, k)
    w = 1.0 / (d ** power + eps)
    w = w / w.sum(axis=1, keepdims=True)
    vals = v[idx]
    return np.einsum("ij,ij->i", w, vals)


__all__ = ["knn_interpolate", "idw_interpolate"]
