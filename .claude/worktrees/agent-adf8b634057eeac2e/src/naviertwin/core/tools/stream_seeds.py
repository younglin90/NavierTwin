"""Streamline seeding — uniform / vorticity-weighted.

Examples:
    >>> import numpy as np
    >>> from naviertwin.core.tools.stream_seeds import uniform_seeds
    >>> seeds = uniform_seeds(bbox=((0, 0), (1, 1)), n=10)
    >>> seeds.shape
    (10, 2)
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def uniform_seeds(
    bbox: tuple[tuple[float, float], tuple[float, float]],
    n: int,
    *,
    seed: int = 0,
) -> NDArray[np.float64]:
    """직사각형 영역에서 균일 무작위 점."""
    rng = np.random.default_rng(seed)
    (x0, y0), (x1, y1) = bbox
    return np.column_stack([
        rng.uniform(x0, x1, n),
        rng.uniform(y0, y1, n),
    ])


def vorticity_weighted_seeds(
    bbox: tuple[tuple[float, float], tuple[float, float]],
    vorticity_field: NDArray[np.float64],
    n: int,
    *,
    seed: int = 0,
) -> NDArray[np.float64]:
    """|ω| 가중 sampling."""
    rng = np.random.default_rng(seed)
    w = np.abs(np.asarray(vorticity_field, dtype=np.float64))
    flat = w.ravel()
    if flat.sum() <= 0:
        return uniform_seeds(bbox, n, seed=seed)
    p = flat / flat.sum()
    idx = rng.choice(flat.size, size=n, p=p)
    nx, ny = w.shape
    ii, jj = np.unravel_index(idx, (nx, ny))
    (x0, y0), (x1, y1) = bbox
    x = x0 + (ii + 0.5) / nx * (x1 - x0)
    y = y0 + (jj + 0.5) / ny * (y1 - y0)
    return np.column_stack([x, y])


__all__ = ["uniform_seeds", "vorticity_weighted_seeds"]
