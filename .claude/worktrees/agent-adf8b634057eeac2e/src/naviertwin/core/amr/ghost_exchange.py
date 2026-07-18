"""Block-AMR ghost cell exchange — 1D 인접 블록 fill.

Examples:
    >>> import numpy as np
    >>> from naviertwin.core.amr.ghost_exchange import exchange_ghost_1d
    >>> blocks = [np.array([1., 2, 3, 0, 0]), np.array([0, 0, 4., 5, 6])]
    >>> ng = 2
    >>> exchange_ghost_1d(blocks, n_ghost=ng)
    >>> blocks[0][-2:].tolist(), blocks[1][:2].tolist()
    ([4.0, 5.0], [2.0, 3.0])
"""

from __future__ import annotations

from collections.abc import Sequence

from numpy.typing import NDArray

from naviertwin._native import _kernels


def exchange_ghost_1d(
    blocks: Sequence[NDArray], *, n_ghost: int = 2,
) -> None:
    """In-place: blocks[i][-ng:] ← blocks[i+1] interior; blocks[i+1][:ng] ← blocks[i]."""
    if _kernels is None:
        raise ImportError("NavierTwin native kernels are required by exchange_ghost_1d")
    _kernels.exchange_ghost_1d(blocks, n_ghost)


__all__ = ["exchange_ghost_1d"]
