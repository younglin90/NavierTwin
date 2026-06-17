"""Bridson Poisson-disk sampling — 최소간격 r 로 blue-noise 샘플.

Examples:
    >>> from naviertwin.core.sampling.poisson_disk import poisson_disk_2d
    >>> pts = poisson_disk_2d(1.0, 1.0, r=0.1, seed=0)
    >>> pts.shape[1]
    2
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def poisson_disk_2d(
    Lx: float, Ly: float, r: float,
    *, k: int = 30, seed: int | None = 0,
) -> NDArray[np.float64]:
    rng = np.random.default_rng(seed)
    cell = r / np.sqrt(2)
    gx = int(np.ceil(Lx / cell))
    gy = int(np.ceil(Ly / cell))
    grid = -np.ones((gx, gy), dtype=np.int64)

    def grid_idx(p):
        return int(p[0] / cell), int(p[1] / cell)

    def fits(p):
        gi, gj = grid_idx(p)
        idx = grid[max(0, gi - 2) : min(gx, gi + 3), max(0, gj - 2) : min(gy, gj + 3)]
        idx = idx[idx >= 0]
        if idx.size == 0:
            return True
        nearby = np.asarray(points, dtype=np.float64)[idx]
        return bool(np.all(np.linalg.norm(nearby - p, axis=1) >= r))

    p0 = np.array([rng.uniform(0, Lx), rng.uniform(0, Ly)])
    points: list[NDArray] = [p0]
    gi, gj = grid_idx(p0)
    grid[gi, gj] = 0
    active = [0]

    while active:
        idx = rng.integers(0, len(active))
        i = active[idx]
        found = False
        trial = 0
        while trial < k:
            theta = rng.uniform(0, 2 * np.pi)
            rr = rng.uniform(r, 2 * r)
            cand = points[i] + np.array([rr * np.cos(theta), rr * np.sin(theta)])
            if 0 <= cand[0] < Lx and 0 <= cand[1] < Ly and fits(cand):
                points.append(cand)
                gi, gj = grid_idx(cand)
                grid[gi, gj] = len(points) - 1
                active.append(len(points) - 1)
                found = True
                break
            trial += 1
        if not found:
            active.pop(idx)
    return np.asarray(points)


__all__ = ["poisson_disk_2d"]
