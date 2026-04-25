"""Vector glyph (arrow) — generate line segments for arrows.

Examples:
    >>> import numpy as np
    >>> from naviertwin.core.visualization.glyph import arrow_segments
    >>> pts = np.array([[0., 0]])
    >>> vec = np.array([[1., 0]])
    >>> segs = arrow_segments(pts, vec, scale=1.0)
    >>> segs.shape  # (n_arrows, 2_endpoints, 2_dim) for shaft only
    (1, 2, 2)
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def arrow_segments(
    points: NDArray[np.float64],
    vectors: NDArray[np.float64],
    *,
    scale: float = 1.0,
) -> NDArray[np.float64]:
    """Returns (N, 2, dim) — start, end pairs."""
    p = np.asarray(points, dtype=np.float64)
    v = np.asarray(vectors, dtype=np.float64)
    end = p + scale * v
    return np.stack([p, end], axis=1)


__all__ = ["arrow_segments"]
