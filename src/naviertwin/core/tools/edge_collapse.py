"""Edge collapse — 가장 짧은 edge 부터 점진적 합치기 (mesh decimation lite).

Examples:
    >>> import numpy as np
    >>> from naviertwin.core.tools.edge_collapse import edge_collapse_once
    >>> v = np.array([[0., 0], [0.1, 0], [1., 0], [0.5, 1]])
    >>> tri = np.array([[0, 1, 3], [1, 2, 3]])
    >>> v2, tri2 = edge_collapse_once(v, tri)
    >>> len(v2) <= len(v)
    True
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def edge_collapse_once(
    verts: NDArray[np.float64], triangles: NDArray[np.int_],
) -> tuple[NDArray[np.float64], NDArray[np.int_]]:
    """가장 짧은 edge 를 midpoint 로 합치기 (1회)."""
    v = np.asarray(verts, dtype=np.float64)
    tri = np.asarray(triangles)
    edge_list = np.vstack([tri[:, [0, 1]], tri[:, [1, 2]], tri[:, [2, 0]]]).astype(int)
    edge_list = np.unique(np.sort(edge_list, axis=1), axis=0)
    lens = np.linalg.norm(v[edge_list[:, 0]] - v[edge_list[:, 1]], axis=1)
    idx = int(np.argmin(lens))
    a, b = edge_list[idx]
    # collapse b → a, midpoint
    v_new = v.copy()
    v_new[a] = 0.5 * (v[a] + v[b])
    # remap b to a
    remap = np.arange(len(v))
    remap[b] = a
    tri_new = remap[tri]
    # drop degenerate triangles (any two equal)
    keep = (
        (tri_new[:, 0] != tri_new[:, 1])
        & (tri_new[:, 1] != tri_new[:, 2])
        & (tri_new[:, 2] != tri_new[:, 0])
    )
    tri_new = tri_new[keep]
    # remove unused vertex b
    used = np.zeros(len(v_new), dtype=bool)
    used[tri_new.ravel()] = True
    keep_v = np.where(used)[0]
    new_idx = -np.ones(len(v_new), dtype=int)
    new_idx[keep_v] = np.arange(len(keep_v))
    return v_new[keep_v], new_idx[tri_new]


__all__ = ["edge_collapse_once"]
