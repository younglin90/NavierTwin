"""Vector glyph (arrow) — generate arrow line segments.

Examples:
    >>> import numpy as np
    >>> from naviertwin.core.visualization.glyph import arrow_segments
    >>> pts = np.array([[0., 0]])
    >>> vec = np.array([[1., 0]])
    >>> segs = arrow_segments(pts, vec, scale=1.0)
    >>> segs.shape
    (1, 2, 2)
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from naviertwin._native import _kernels


def arrow_segments(
    points: NDArray[np.float64],
    vectors: NDArray[np.float64],
    *,
    scale: float = 1.0,
) -> NDArray[np.float64]:
    """Returns (N, 2, dim) — start, end pairs."""
    if _kernels is None:
        raise ImportError("NavierTwin native kernels are required by arrow_segments")
    return _kernels.arrow_segments(
        np.asarray(points, dtype=np.float64),
        np.asarray(vectors, dtype=np.float64),
        scale,
    )


__all__ = ["arrow_segments"]
