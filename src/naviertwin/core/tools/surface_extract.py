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

from naviertwin._native import _kernels


def boundary_faces_tet(tets: NDArray[np.int_]) -> NDArray[np.int_]:
    """tet (M, 4) → boundary triangle faces (K, 3)."""
    if _kernels is None:
        raise ImportError("NavierTwin native kernels are required by boundary_faces_tet")
    return _kernels.boundary_faces_tet(np.asarray(tets, dtype=np.int64))


__all__ = ["boundary_faces_tet"]
