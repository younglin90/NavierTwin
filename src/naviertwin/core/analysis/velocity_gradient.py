"""속도장 기울기 텐서 분해 — S(strain), Ω(rotation), 불변량 Q/R.

Examples:
    >>> import numpy as np
    >>> from naviertwin.core.analysis.velocity_gradient import decompose_J_3x3
    >>> J = np.array([[1,2,3],[0,-1,1],[2,0,1]], dtype=float)
    >>> S, Omega = decompose_J_3x3(J)
    >>> np.allclose(S, S.T)
    True
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from naviertwin._native import _kernels


def decompose_J_3x3(
    J: NDArray[np.float64],
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """J = S + Ω, S = (J + Jᵀ)/2, Ω = (J - Jᵀ)/2."""
    J = np.asarray(J, dtype=np.float64)
    if _kernels is None:
        return _decompose_J_3x3_numpy(J)
    return _kernels.decompose_j_3x3(J)


def invariants_3x3(J: NDArray[np.float64]) -> dict[str, float]:
    """속도장 gradient 불변량 — P, Q, R (nondiv flow 이면 P=0).

    P = -tr J, Q = ½(P² - tr(J²)), R = -det J.
    """
    J = np.asarray(J, dtype=np.float64)
    if _kernels is None:
        return _invariants_3x3_numpy(J)
    return dict(_kernels.invariants_3x3(J))


def field_J_2d(
    u: NDArray[np.float64], v: NDArray[np.float64],
    dx: float = 1.0, dy: float = 1.0,
) -> NDArray[np.float64]:
    """격자의 2D velocity gradient J (ny, nx, 2, 2)."""
    if _kernels is None:
        return _field_J_2d_numpy(u, v, dx, dy)
    return _kernels.field_j_2d(u, v, dx, dy)


def _decompose_J_3x3_numpy(
    J: NDArray[np.float64],
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """NumPy fallback matching ``decompose_J_3x3``."""
    J = _as_3x3(J)
    return 0.5 * (J + J.T), 0.5 * (J - J.T)


def _invariants_3x3_numpy(J: NDArray[np.float64]) -> dict[str, float]:
    """NumPy fallback matching ``invariants_3x3``."""
    J = _as_3x3(J)
    trace = float(np.trace(J))
    p = -trace
    q = 0.5 * (p * p - float(np.trace(J @ J)))
    r = -_det_3x3(J)
    return {"P": p, "Q": q, "R": r}


def _field_J_2d_numpy(
    u: NDArray[np.float64], v: NDArray[np.float64],
    dx: float = 1.0, dy: float = 1.0,
) -> NDArray[np.float64]:
    """NumPy fallback matching the native 2D gradient stencil."""
    u_arr, v_arr = _check_same_2d(u, v)
    _check_spacing(dx, dy)

    du_dx = _grad_x(u_arr, dx)
    du_dy = _grad_y(u_arr, dy)
    dv_dx = _grad_x(v_arr, dx)
    dv_dy = _grad_y(v_arr, dy)

    out = np.empty((*u_arr.shape, 2, 2), dtype=np.float64)
    out[..., 0, 0] = du_dx
    out[..., 0, 1] = du_dy
    out[..., 1, 0] = dv_dx
    out[..., 1, 1] = dv_dy
    return out


def _as_3x3(J: NDArray[np.float64]) -> NDArray[np.float64]:
    arr = np.asarray(J, dtype=np.float64)
    if arr.shape != (3, 3):
        raise ValueError("3x3 expected")
    return arr


def _det_3x3(J: NDArray[np.float64]) -> float:
    return float(
        J[0, 0] * (J[1, 1] * J[2, 2] - J[1, 2] * J[2, 1])
        - J[0, 1] * (J[1, 0] * J[2, 2] - J[1, 2] * J[2, 0])
        + J[0, 2] * (J[1, 0] * J[2, 1] - J[1, 1] * J[2, 0])
    )


def _check_same_2d(
    u: NDArray[np.float64], v: NDArray[np.float64],
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    u_arr = np.asarray(u, dtype=np.float64)
    v_arr = np.asarray(v, dtype=np.float64)
    if u_arr.ndim != 2 or v_arr.ndim != 2:
        raise ValueError("2D arrays expected")
    if u_arr.shape != v_arr.shape:
        raise ValueError("u and v must have the same shape")
    if u_arr.shape[0] < 2 or u_arr.shape[1] < 2:
        raise ValueError("each array axis must have at least 2 points")
    return u_arr, v_arr


def _check_spacing(dx: float, dy: float) -> None:
    if dx == 0.0 or dy == 0.0:
        raise ValueError("dx and dy must be non-zero")


def _grad_x(a: NDArray[np.float64], dx: float) -> NDArray[np.float64]:
    out = np.empty(a.shape, dtype=np.float64)
    out[:, 0] = (a[:, 1] - a[:, 0]) / dx
    out[:, -1] = (a[:, -1] - a[:, -2]) / dx
    if a.shape[1] > 2:
        out[:, 1:-1] = (a[:, 2:] - a[:, :-2]) / (2.0 * dx)
    return out


def _grad_y(a: NDArray[np.float64], dy: float) -> NDArray[np.float64]:
    out = np.empty(a.shape, dtype=np.float64)
    out[0, :] = (a[1, :] - a[0, :]) / dy
    out[-1, :] = (a[-1, :] - a[-2, :]) / dy
    if a.shape[0] > 2:
        out[1:-1, :] = (a[2:, :] - a[:-2, :]) / (2.0 * dy)
    return out


__all__ = ["decompose_J_3x3", "invariants_3x3", "field_J_2d"]
