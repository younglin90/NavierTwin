"""정규 격자 고차 정확도 유한 차분 — 그래디언트, 발산, 회전, 라플라시안.

uniform grid 후처리 표준. 2차/4차 중앙 차분; 경계는 자동 자릿수 강하.

상용 툴 대응:
    - Tecplot 360: Calculate Variables (Gradient, Divergence, Curl)
    - Ansys CFD-Post: Vector Field Variables
    - ParaView: Compute Derivatives filter

Examples:
    >>> import numpy as np
    >>> # f(x, y) = x² + y² → gradient = (2x, 2y)
    >>> y = x = np.linspace(0, 1, 50)
    >>> X, Y = np.meshgrid(x, y, indexing="ij")
    >>> f = X**2 + Y**2
    >>> from naviertwin.core.flow_analysis.grid_derivatives import gradient_2d
    >>> gx, gy = gradient_2d(f, dx=x[1]-x[0], dy=y[1]-y[0], order=4)
    >>> abs(gx[25, 25] - 2*X[25, 25]) < 1e-3
    True
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from naviertwin.utils.logger import get_logger

logger = get_logger(__name__)


def _central_diff_axis(
    f: NDArray[np.float64],
    h: float,
    axis: int,
    order: int = 2,
) -> NDArray[np.float64]:
    """주어진 축에 대해 한 번의 중앙 차분."""
    if order not in (2, 4):
        raise ValueError(f"order must be 2 or 4, got {order}")
    if h <= 0:
        raise ValueError(f"h must be > 0, got {h}")

    n = f.shape[axis]
    if order == 2:
        if n < 3:
            raise ValueError(f"need at least 3 points along axis {axis}")
        out = np.zeros_like(f)
        # 중앙: (f[i+1] - f[i-1]) / (2h)
        sl_p = [slice(None)] * f.ndim
        sl_m = [slice(None)] * f.ndim
        sl_p[axis] = slice(2, n)
        sl_m[axis] = slice(0, n - 2)
        center = (f[tuple(sl_p)] - f[tuple(sl_m)]) / (2.0 * h)
        sl_c = [slice(None)] * f.ndim
        sl_c[axis] = slice(1, n - 1)
        out[tuple(sl_c)] = center
        # 경계 1차
        sl0 = [slice(None)] * f.ndim
        sl1 = [slice(None)] * f.ndim
        sl0[axis] = 0
        sl1[axis] = 1
        out[tuple(sl0)] = (f[tuple(sl1)] - f[tuple(sl0)]) / h
        sl_n1 = [slice(None)] * f.ndim
        sl_n2 = [slice(None)] * f.ndim
        sl_n1[axis] = n - 1
        sl_n2[axis] = n - 2
        out[tuple(sl_n1)] = (f[tuple(sl_n1)] - f[tuple(sl_n2)]) / h
        return out

    # 4차 중앙 차분: (-f[i+2] + 8 f[i+1] - 8 f[i-1] + f[i-2]) / (12 h)
    if n < 5:
        # 4차 불가 → 2차 폴백
        return _central_diff_axis(f, h, axis, order=2)
    out = np.zeros_like(f)
    sl_pp = [slice(None)] * f.ndim
    sl_p = [slice(None)] * f.ndim
    sl_m = [slice(None)] * f.ndim
    sl_mm = [slice(None)] * f.ndim
    sl_c = [slice(None)] * f.ndim
    sl_pp[axis] = slice(4, n)
    sl_p[axis] = slice(3, n - 1)
    sl_m[axis] = slice(1, n - 3)
    sl_mm[axis] = slice(0, n - 4)
    sl_c[axis] = slice(2, n - 2)
    out[tuple(sl_c)] = (
        -f[tuple(sl_pp)] + 8.0 * f[tuple(sl_p)] - 8.0 * f[tuple(sl_m)] + f[tuple(sl_mm)]
    ) / (12.0 * h)

    # 경계는 2차 차분으로 보정
    bnd = _central_diff_axis(f, h, axis, order=2)
    sl_b1 = [slice(None)] * f.ndim
    sl_b2 = [slice(None)] * f.ndim
    sl_b1[axis] = slice(0, 2)
    sl_b2[axis] = slice(n - 2, n)
    out[tuple(sl_b1)] = bnd[tuple(sl_b1)]
    out[tuple(sl_b2)] = bnd[tuple(sl_b2)]
    return out


def gradient_2d(
    f: NDArray[np.float64],
    dx: float = 1.0,
    dy: float = 1.0,
    order: int = 2,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """2D 스칼라장 그래디언트.

    Args:
        f: (Nx, Ny) 스칼라장 (인덱스 [i, j] = (x_i, y_j)).
        dx, dy: 격자 간격.
        order: 정확도 차수 (2 또는 4).

    Returns:
        (∂f/∂x, ∂f/∂y).

    Raises:
        ValueError: f가 2D가 아닌 경우.
    """
    f = np.asarray(f, dtype=np.float64)
    if f.ndim != 2:
        raise ValueError(f"f must be 2D, got {f.shape}")
    fx = _central_diff_axis(f, dx, axis=0, order=order)
    fy = _central_diff_axis(f, dy, axis=1, order=order)
    return fx, fy


def gradient_3d(
    f: NDArray[np.float64],
    dx: float = 1.0,
    dy: float = 1.0,
    dz: float = 1.0,
    order: int = 2,
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """3D 스칼라장 그래디언트.

    Args:
        f: (Nx, Ny, Nz) 스칼라장.
        dx, dy, dz: 격자 간격.
        order: 차수.

    Returns:
        (∂f/∂x, ∂f/∂y, ∂f/∂z).
    """
    f = np.asarray(f, dtype=np.float64)
    if f.ndim != 3:
        raise ValueError(f"f must be 3D, got {f.shape}")
    fx = _central_diff_axis(f, dx, axis=0, order=order)
    fy = _central_diff_axis(f, dy, axis=1, order=order)
    fz = _central_diff_axis(f, dz, axis=2, order=order)
    return fx, fy, fz


def divergence_3d(
    u: NDArray[np.float64],
    v: NDArray[np.float64],
    w: NDArray[np.float64],
    dx: float = 1.0,
    dy: float = 1.0,
    dz: float = 1.0,
    order: int = 2,
) -> NDArray[np.float64]:
    """3D 벡터장 발산 ∇·U = ∂u/∂x + ∂v/∂y + ∂w/∂z.

    Args:
        u, v, w: (Nx, Ny, Nz) 속도 성분.
        dx, dy, dz, order: 격자/차수.

    Returns:
        (Nx, Ny, Nz) 발산 필드.

    Raises:
        ValueError: u/v/w 형상 불일치.
    """
    u = np.asarray(u, dtype=np.float64)
    v = np.asarray(v, dtype=np.float64)
    w = np.asarray(w, dtype=np.float64)
    if not (u.shape == v.shape == w.shape) or u.ndim != 3:
        raise ValueError(
            f"u/v/w must be same-shape 3D, got {u.shape}, {v.shape}, {w.shape}"
        )
    ux = _central_diff_axis(u, dx, axis=0, order=order)
    vy = _central_diff_axis(v, dy, axis=1, order=order)
    wz = _central_diff_axis(w, dz, axis=2, order=order)
    return ux + vy + wz


def curl_3d(
    u: NDArray[np.float64],
    v: NDArray[np.float64],
    w: NDArray[np.float64],
    dx: float = 1.0,
    dy: float = 1.0,
    dz: float = 1.0,
    order: int = 2,
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """3D 회전 ω = ∇×U.

    ω_x = ∂w/∂y - ∂v/∂z
    ω_y = ∂u/∂z - ∂w/∂x
    ω_z = ∂v/∂x - ∂u/∂y

    Args:
        u, v, w: 속도 성분.
        dx, dy, dz, order: 격자/차수.

    Returns:
        (ω_x, ω_y, ω_z).
    """
    u = np.asarray(u, dtype=np.float64)
    v = np.asarray(v, dtype=np.float64)
    w = np.asarray(w, dtype=np.float64)
    if not (u.shape == v.shape == w.shape) or u.ndim != 3:
        raise ValueError("u/v/w must be same-shape 3D arrays")
    wy = _central_diff_axis(w, dy, axis=1, order=order)
    vz = _central_diff_axis(v, dz, axis=2, order=order)
    uz = _central_diff_axis(u, dz, axis=2, order=order)
    wx = _central_diff_axis(w, dx, axis=0, order=order)
    vx = _central_diff_axis(v, dx, axis=0, order=order)
    uy = _central_diff_axis(u, dy, axis=1, order=order)
    return wy - vz, uz - wx, vx - uy


def laplacian_2d(
    f: NDArray[np.float64],
    dx: float = 1.0,
    dy: float = 1.0,
    order: int = 2,
) -> NDArray[np.float64]:
    """2D 라플라시안 ∇²f.

    Args:
        f: (Nx, Ny) 스칼라장.
        dx, dy: 격자.
        order: 차수.

    Returns:
        (Nx, Ny) ∇²f.

    Raises:
        ValueError: f가 2D 아님.
    """
    f = np.asarray(f, dtype=np.float64)
    if f.ndim != 2:
        raise ValueError(f"f must be 2D, got {f.shape}")
    fx = _central_diff_axis(f, dx, axis=0, order=order)
    fy = _central_diff_axis(f, dy, axis=1, order=order)
    fxx = _central_diff_axis(fx, dx, axis=0, order=order)
    fyy = _central_diff_axis(fy, dy, axis=1, order=order)
    return fxx + fyy


def laplacian_3d(
    f: NDArray[np.float64],
    dx: float = 1.0,
    dy: float = 1.0,
    dz: float = 1.0,
    order: int = 2,
) -> NDArray[np.float64]:
    """3D 라플라시안 ∇²f."""
    f = np.asarray(f, dtype=np.float64)
    if f.ndim != 3:
        raise ValueError(f"f must be 3D, got {f.shape}")
    fx = _central_diff_axis(f, dx, axis=0, order=order)
    fxx = _central_diff_axis(fx, dx, axis=0, order=order)
    fy = _central_diff_axis(f, dy, axis=1, order=order)
    fyy = _central_diff_axis(fy, dy, axis=1, order=order)
    fz = _central_diff_axis(f, dz, axis=2, order=order)
    fzz = _central_diff_axis(fz, dz, axis=2, order=order)
    return fxx + fyy + fzz


__all__ = [
    "gradient_2d",
    "gradient_3d",
    "divergence_3d",
    "curl_3d",
    "laplacian_2d",
    "laplacian_3d",
]
