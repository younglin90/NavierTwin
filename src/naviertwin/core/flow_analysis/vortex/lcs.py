"""Lagrangian Coherent Structures (LCS) via FTLE.

Finite-Time Lyapunov Exponent field 계산. 2D 정상/비정상 속도장에서
입자 궤적을 RK4 로 T 시간 적분한 뒤 flow-map Jacobian 의 최대 특이값을 통해
FTLE = (1/|T|) log(max SV).

Examples:
    >>> import numpy as np
    >>> from naviertwin.core.flow_analysis.vortex.lcs import compute_ftle_2d
    >>> # 정상 Double-gyre-like 속도장
    >>> def U(t, x, y):
    ...     return -np.sin(np.pi * x) * np.cos(np.pi * y)
    >>> def V(t, x, y):
    ...     return np.cos(np.pi * x) * np.sin(np.pi * y)
    >>> ftle = compute_ftle_2d(U, V, nx=20, ny=20, T=2.0, dt=0.1)
    >>> ftle.shape
    (20, 20)
"""

from __future__ import annotations

from typing import Callable

import numpy as np
from numpy.typing import NDArray

from naviertwin._native import _kernels
from naviertwin.utils.logger import get_logger

logger = get_logger(__name__)


def compute_ftle_2d(
    u_fn: Callable[[float, NDArray, NDArray], NDArray],
    v_fn: Callable[[float, NDArray, NDArray], NDArray],
    nx: int = 50,
    ny: int = 50,
    x_range: tuple[float, float] = (0.0, 1.0),
    y_range: tuple[float, float] = (0.0, 1.0),
    T: float = 1.0,
    dt: float = 0.05,
    t0: float = 0.0,
) -> NDArray[np.float64]:
    """2D FTLE 필드를 계산한다.

    Args:
        u_fn, v_fn: 속도 성분 함수 (t, x, y) → scalar/array.
        nx, ny: 격자 해상도.
        x_range, y_range: 도메인 범위.
        T: 적분 시간 (양수 forward / 음수 backward).
        dt: 시간 스텝.
        t0: 시작 시각.

    Returns:
        (ny, nx) FTLE 필드.
    """
    if _kernels is None:
        raise ImportError("naviertwin._native._kernels is required by compute_ftle_2d")
    x = np.linspace(*x_range, nx)
    y = np.linspace(*y_range, ny)
    Xg, Yg = np.meshgrid(x, y)
    X = Xg.copy()
    Y = Yg.copy()

    n_steps = int(abs(T) / dt)
    sign = np.sign(T) if T != 0 else 1.0

    k = 0
    while k < n_steps:
        t = t0 + sign * k * dt
        k1u = u_fn(t, X, Y)
        k1v = v_fn(t, X, Y)
        k2u = u_fn(t + sign * dt / 2, X + sign * dt / 2 * k1u, Y + sign * dt / 2 * k1v)
        k2v = v_fn(t + sign * dt / 2, X + sign * dt / 2 * k1u, Y + sign * dt / 2 * k1v)
        k3u = u_fn(t + sign * dt / 2, X + sign * dt / 2 * k2u, Y + sign * dt / 2 * k2v)
        k3v = v_fn(t + sign * dt / 2, X + sign * dt / 2 * k2u, Y + sign * dt / 2 * k2v)
        k4u = u_fn(t + sign * dt, X + sign * dt * k3u, Y + sign * dt * k3v)
        k4v = v_fn(t + sign * dt, X + sign * dt * k3u, Y + sign * dt * k3v)
        X = X + sign * dt / 6 * (k1u + 2 * k2u + 2 * k3u + k4u)
        Y = Y + sign * dt / 6 * (k1v + 2 * k2v + 2 * k3v + k4v)
        k += 1

    dx = x[1] - x[0]
    dy = y[1] - y[0]
    ftle = _kernels.lcs_ftle_from_flow_map(X, Y, dx, dy, T)
    logger.info("FTLE 계산 완료: grid=%dx%d, T=%.3g", nx, ny, T)
    return ftle


__all__ = ["compute_ftle_2d"]
