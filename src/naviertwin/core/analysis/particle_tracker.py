"""Lagrangian particle tracker — 속도장에서 RK4 로 입자 궤적 적분.

Examples:
    >>> import numpy as np
    >>> from naviertwin.core.analysis.particle_tracker import track_particles_2d
    >>> u_field = np.ones((10, 10))  # 균일 속도
    >>> v_field = np.zeros((10, 10))
    >>> seeds = np.array([[0.5, 0.5]])
    >>> trails = track_particles_2d(u_field, v_field, seeds, Lx=1.0, Ly=1.0, dt=0.01, n_steps=10)
    >>> trails.shape
    (1, 11, 2)
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from naviertwin._native import _kernels

if _kernels is None:  # pragma: no cover
    raise ImportError("NavierTwin native kernels are required by particle tracking")


def track_particles_2d(
    u: NDArray[np.float64], v: NDArray[np.float64],
    seeds: NDArray[np.float64],
    *, Lx: float = 1.0, Ly: float = 1.0,
    dt: float = 0.01, n_steps: int = 100,
) -> NDArray[np.float64]:
    """(n_seeds, n_steps+1, 2) 궤적."""
    return _kernels.track_particles_2d(
        np.asarray(u, dtype=np.float64),
        np.asarray(v, dtype=np.float64),
        np.asarray(seeds, dtype=np.float64),
        float(Lx),
        float(Ly),
        float(dt),
        int(n_steps),
    )


def residence_time(
    trails: NDArray[np.float64], *, box: tuple[float, float, float, float],
    dt: float,
) -> NDArray[np.float64]:
    """particle 별 box 내 머문 시간."""
    xmin, ymin, xmax, ymax = box
    mask = (
        (trails[:, :, 0] >= xmin) & (trails[:, :, 0] <= xmax)
        & (trails[:, :, 1] >= ymin) & (trails[:, :, 1] <= ymax)
    )
    return mask.sum(axis=1) * dt


__all__ = ["track_particles_2d", "residence_time"]
