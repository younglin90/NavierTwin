"""scipy.spatial.cKDTree 래퍼 — 대용량 KNN/radius/ball 검색.

Examples:
    >>> import numpy as np
    >>> from naviertwin.core.analysis.kdtree_wrapper import KDTreeIndex
    >>> rng = np.random.default_rng(0)
    >>> pts = rng.standard_normal((1000, 3))
    >>> kd = KDTreeIndex(pts)
    >>> d, idx = kd.knn(np.array([[0., 0., 0.]]), k=5)
    >>> idx.shape
    (1, 5)
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


class KDTreeIndex:
    def __init__(self, points: NDArray[np.float64]) -> None:
        try:
            from scipy.spatial import cKDTree
        except ImportError as exc:
            raise RuntimeError("scipy 필요") from exc
        self.points = np.asarray(points, dtype=np.float64)
        self._tree = cKDTree(self.points)

    def knn(self, queries: NDArray[np.float64], k: int = 1) -> tuple[NDArray, NDArray]:
        """(n_q, k) 거리/인덱스."""
        q = np.atleast_2d(queries)
        d, i = self._tree.query(q, k=k)
        if k == 1:
            d = d[:, None]
            i = i[:, None]
        return d, i

    def radius(self, q: NDArray[np.float64], r: float) -> NDArray[np.int64]:
        """반경 r 내 모든 점 인덱스."""
        q = np.asarray(q, dtype=np.float64).ravel()
        return np.asarray(self._tree.query_ball_point(q, r), dtype=np.int64)

    def pairs_within(self, r: float) -> list[tuple[int, int]]:
        """자기 자신 내에서 거리 ≤ r 인 점 쌍."""
        return list(self._tree.query_pairs(r))

    def count_within(self, q: NDArray[np.float64], r: float) -> int:
        return int(self._tree.query_ball_point(q, r, return_length=True))


__all__ = ["KDTreeIndex"]
