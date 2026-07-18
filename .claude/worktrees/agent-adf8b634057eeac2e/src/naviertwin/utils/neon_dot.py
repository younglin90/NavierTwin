"""ARM NEON dot proxy — uses numpy as a stand-in to emulate SIMD-accelerated dot.

Provides a name-stable API so embedded paths may swap to true NEON later.

Examples:
    >>> import numpy as np
    >>> from naviertwin.utils.neon_dot import dot_int8
    >>> dot_int8(np.array([1, 2, 3], dtype=np.int8), np.array([4, 5, 6], dtype=np.int8))
    32
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def dot_int8(a: NDArray[np.int8], b: NDArray[np.int8]) -> int:
    """Dot product with int8 vectors and int32 accumulator."""
    return int(np.dot(a.astype(np.int32), b.astype(np.int32)))


def dot_f32(a: NDArray[np.float32], b: NDArray[np.float32]) -> float:
    return float(np.dot(a.astype(np.float32), b.astype(np.float32)))


__all__ = ["dot_f32", "dot_int8"]
