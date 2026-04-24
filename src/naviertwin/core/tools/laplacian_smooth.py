"""Laplacian mesh smoothing — 각 정점을 인접 정점 평균으로 이동.

Examples:
    >>> import numpy as np
    >>> from naviertwin.core.tools.laplacian_smooth import laplacian_smooth
    >>> verts = np.array([[0., 0.], [1., 0.], [2., 0.], [1., 0.5]])
    >>> edges = [(0, 1), (1, 2), (1, 3)]
    >>> v2 = laplacian_smooth(verts, edges, n_iter=1, fixed=[0, 2])
    >>> v2.shape
    (4, 2)
"""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np
from numpy.typing import NDArray


def laplacian_smooth(
    verts: NDArray[np.float64],
    edges: Iterable[tuple[int, int]],
    *,
    n_iter: int = 5,
    alpha: float = 0.5,
    fixed: Iterable[int] | None = None,
) -> NDArray[np.float64]:
    """v_i ← (1-α) v_i + α mean(v_j : j ∈ N(i))."""
    v = np.asarray(verts, dtype=np.float64).copy()
    n = v.shape[0]
    adj: list[list[int]] = [[] for _ in range(n)]
    for a, b in edges:
        adj[a].append(b)
        adj[b].append(a)
    fixed_set = set(fixed) if fixed is not None else set()
    for _ in range(n_iter):
        v_new = v.copy()
        for i in range(n):
            if i in fixed_set or not adj[i]:
                continue
            mean = v[adj[i]].mean(axis=0)
            v_new[i] = (1 - alpha) * v[i] + alpha * mean
        v = v_new
    return v


__all__ = ["laplacian_smooth"]
