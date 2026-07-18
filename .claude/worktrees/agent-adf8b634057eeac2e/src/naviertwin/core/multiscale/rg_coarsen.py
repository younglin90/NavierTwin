"""Renormalization-group coarsening 1D — block-spin majority rule.

Examples:
    >>> import numpy as np
    >>> from naviertwin.core.multiscale.rg_coarsen import block_spin
    >>> s = np.array([1, 1, -1, 1, -1, -1])
    >>> block_spin(s, block=2).tolist()
    [1, -1, -1]
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def block_spin(spins: NDArray[np.int_], block: int = 2) -> NDArray[np.int_]:
    s = np.asarray(spins, dtype=int)
    n = (len(s) // block) * block
    blocks = s[:n].reshape(-1, block)
    sums = blocks.sum(axis=1)
    out = np.where(sums >= 0, 1, -1)
    # tie (sum=0): take first spin in block
    ties = (sums == 0)
    out = np.where(ties, blocks[:, 0], out)
    return out


def rg_iterate(spins: NDArray[np.int_], *, block: int = 2, n_iter: int = 1) -> NDArray[np.int_]:
    s = spins
    step = 0
    while step < n_iter:
        s = block_spin(s, block)
        step += 1
    return s


__all__ = ["block_spin", "rg_iterate"]
