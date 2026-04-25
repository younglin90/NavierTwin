"""Train/val/test split with stratify.

Examples:
    >>> import numpy as np
    >>> from naviertwin.utils.dataset_split import stratified_split
    >>> y = np.array([0]*8 + [1]*4)
    >>> i_tr, i_va, i_te = stratified_split(y, ratios=(0.5, 0.25, 0.25), seed=0)
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def stratified_split(
    y: NDArray, *, ratios: tuple[float, float, float] = (0.7, 0.15, 0.15),
    seed: int = 0,
) -> tuple[NDArray, NDArray, NDArray]:
    rng = np.random.default_rng(seed)
    y = np.asarray(y)
    classes = np.unique(y)
    tr, va, te = [], [], []
    for c in classes:
        idx = np.where(y == c)[0]
        rng.shuffle(idx)
        n = len(idx)
        n_tr = int(round(n * ratios[0]))
        n_va = int(round(n * ratios[1]))
        tr.extend(idx[:n_tr].tolist())
        va.extend(idx[n_tr:n_tr + n_va].tolist())
        te.extend(idx[n_tr + n_va:].tolist())
    return np.array(sorted(tr)), np.array(sorted(va)), np.array(sorted(te))


__all__ = ["stratified_split"]
