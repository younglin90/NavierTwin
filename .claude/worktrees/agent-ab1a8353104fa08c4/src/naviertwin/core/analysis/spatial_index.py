"""간단한 공간 색인 — grid bucket (빠른 근접 점 검색).

Examples:
    >>> import numpy as np
    >>> from naviertwin.core.analysis.spatial_index import GridIndex
    >>> pts = np.random.default_rng(0).standard_normal((1000, 3))
    >>> idx = GridIndex(pts, cell_size=0.5)
    >>> idx.query_radius(np.array([0., 0., 0.]), 0.3).size >= 0
    True
"""

from __future__ import annotations

from collections import defaultdict

import numpy as np
from numpy.typing import NDArray


class GridIndex:
    """균일 셀 공간 색인 — O(1) 평균 근접 질의."""

    def __init__(
        self, points: NDArray[np.float64], cell_size: float,
    ) -> None:
        self.points = np.asarray(points, dtype=np.float64)
        if self.points.ndim != 2:
            raise ValueError("points (n, d) 필요")
        self.cell = float(cell_size)
        self._buckets: dict[tuple[int, ...], list[int]] = defaultdict(list)
        keys = np.floor(self.points / self.cell).astype(np.int64)
        i = 0
        while i < keys.shape[0]:
            self._buckets[tuple(keys[i])].append(i)
            i += 1

    def query_radius(
        self, q: NDArray[np.float64], radius: float,
    ) -> NDArray[np.int64]:
        q = np.asarray(q, dtype=np.float64).ravel()
        d = self.points.shape[1]
        span = int(np.ceil(radius / self.cell))
        kq = np.floor(q / self.cell).astype(np.int64)
        hits: list[int] = []
        # 주변 셀 iterate
        deltas = np.array(list(np.ndindex(*([2 * span + 1] * d))), dtype=np.int64)
        delta_idx = 0
        while delta_idx < deltas.shape[0]:
            key = tuple(kq + deltas[delta_idx] - span)
            bucket = self._buckets.get(key, None)
            if bucket:
                hits.extend(bucket)
            delta_idx += 1
        if not hits:
            return np.array([], dtype=np.int64)
        hits_arr = np.asarray(hits, dtype=np.int64)
        dists = np.linalg.norm(self.points[hits_arr] - q, axis=1)
        return hits_arr[dists <= radius]

    def nearest(self, q: NDArray[np.float64]) -> int:
        """가장 가까운 점의 인덱스."""
        # 셀이 비어있을 수도 있으므로 반경 확장
        r = self.cell
        attempt = 0
        while attempt < 20:
            idx = self.query_radius(q, r)
            if idx.size > 0:
                dists = np.linalg.norm(self.points[idx] - q, axis=1)
                return int(idx[np.argmin(dists)])
            r *= 2
            attempt += 1
        # fallback
        dists = np.linalg.norm(self.points - q, axis=1)
        return int(np.argmin(dists))


__all__ = ["GridIndex"]
