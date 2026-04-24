"""2D/3D 벡터장 스트림라인 — RK4 정적분기.

CFD 사후처리 — 점에서 시작해 속도장을 따라 경로 추적.

Examples:
    >>> import numpy as np
    >>> from naviertwin.core.analysis.streamline import integrate_streamline
    >>> def vf(p): return np.array([-p[1], p[0]])  # 회전장
    >>> pts = integrate_streamline(vf, start=np.array([1.0, 0.0]), dt=0.01, n_steps=100)
    >>> pts.shape
    (101, 2)
"""

from __future__ import annotations

from typing import Callable

import numpy as np
from numpy.typing import NDArray


def rk4_step(
    vf: Callable[[NDArray[np.float64]], NDArray[np.float64]],
    p: NDArray[np.float64],
    dt: float,
) -> NDArray[np.float64]:
    k1 = vf(p)
    k2 = vf(p + 0.5 * dt * k1)
    k3 = vf(p + 0.5 * dt * k2)
    k4 = vf(p + dt * k3)
    return p + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)


def integrate_streamline(
    vf: Callable[[NDArray[np.float64]], NDArray[np.float64]],
    start: NDArray[np.float64],
    dt: float,
    n_steps: int,
    *,
    stop_if_stagnant: float = 1e-12,
) -> NDArray[np.float64]:
    """RK4 로 스트림라인 적분. (n_steps+1, dim) 경로 반환."""
    start = np.asarray(start, dtype=np.float64)
    path = np.zeros((n_steps + 1, start.size), dtype=np.float64)
    path[0] = start
    p = start.copy()
    for i in range(n_steps):
        p_next = rk4_step(vf, p, dt)
        if np.linalg.norm(p_next - p) < stop_if_stagnant:
            path[i + 1:] = p_next
            break
        p = p_next
        path[i + 1] = p
    return path


def integrate_streamlines(
    vf: Callable[[NDArray[np.float64]], NDArray[np.float64]],
    starts: NDArray[np.float64],
    dt: float,
    n_steps: int,
) -> NDArray[np.float64]:
    """여러 시드 병렬 적분. starts: (n_seeds, dim)."""
    starts = np.asarray(starts, dtype=np.float64)
    out = np.stack(
        [integrate_streamline(vf, s, dt, n_steps) for s in starts],
        axis=0,
    )
    return out  # (n_seeds, n_steps+1, dim)


__all__ = ["rk4_step", "integrate_streamline", "integrate_streamlines"]
