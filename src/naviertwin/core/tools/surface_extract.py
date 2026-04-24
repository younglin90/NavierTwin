"""Surface extraction — tetra cell 의 boundary 면 추출.

각 face 가 정확히 한 cell 에 속하면 boundary.

Examples:
    >>> import numpy as np
    >>> from naviertwin.core.tools.surface_extract import boundary_faces_tet
    >>> tets = np.array([[0, 1, 2, 3]])
    >>> faces = boundary_faces_tet(tets)
    >>> faces.shape
    (4, 3)
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

_TET_FACES = ((0, 1, 2), (0, 1, 3), (0, 2, 3), (1, 2, 3))


def boundary_faces_tet(tets: NDArray[np.int_]) -> NDArray[np.int_]:
    """tet (M, 4) → boundary triangle faces (K, 3)."""
    tets = np.asarray(tets)
    counts: dict[tuple[int, int, int], int] = {}
    for t in tets:
        for f in _TET_FACES:
            tri = tuple(sorted((int(t[f[0]]), int(t[f[1]]), int(t[f[2]]))))
            counts[tri] = counts.get(tri, 0) + 1
    return np.array([list(k) for k, c in counts.items() if c == 1], dtype=int)


__all__ = ["boundary_faces_tet"]
