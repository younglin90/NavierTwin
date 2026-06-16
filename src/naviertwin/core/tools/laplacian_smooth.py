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

from naviertwin._native import _kernels

if _kernels is None:  # pragma: no cover
    raise ImportError("NavierTwin native kernels are required")


def laplacian_smooth(
    verts: NDArray[np.float64],
    edges: Iterable[tuple[int, int]],
    *,
    n_iter: int = 5,
    alpha: float = 0.5,
    fixed: Iterable[int] | None = None,
) -> NDArray[np.float64]:
    """v_i ← (1-α) v_i + α mean(v_j : j ∈ N(i))."""
    edge_arr = np.asarray(list(edges), dtype=np.int64).reshape(-1, 2)
    fixed_arr = np.asarray([] if fixed is None else list(fixed), dtype=np.int64)
    return _kernels.laplacian_smooth(
        np.asarray(verts, dtype=np.float64),
        edge_arr,
        int(n_iter),
        float(alpha),
        fixed_arr,
    )


__all__ = ["laplacian_smooth"]
