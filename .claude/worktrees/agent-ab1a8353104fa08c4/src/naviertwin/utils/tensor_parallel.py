"""Tensor parallel split — column-/row-parallel weight splits.

Examples:
    >>> import numpy as np
    >>> from naviertwin.utils.tensor_parallel import column_parallel
    >>> W = np.arange(12).reshape(3, 4)
    >>> shards = column_parallel(W, n=2)
    >>> shards[0].shape, shards[1].shape
    ((3, 2), (3, 2))
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def column_parallel(W: NDArray[np.float64], n: int) -> list[NDArray]:
    """W (out, in) → list of (out, in/n)."""
    return list(np.array_split(np.asarray(W), n, axis=1))


def row_parallel(W: NDArray[np.float64], n: int) -> list[NDArray]:
    """W (out, in) → list of (out/n, in)."""
    return list(np.array_split(np.asarray(W), n, axis=0))


def all_reduce_sum(shards: list[NDArray]) -> NDArray:
    return np.sum(shards, axis=0)


__all__ = ["all_reduce_sum", "column_parallel", "row_parallel"]
