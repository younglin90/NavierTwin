"""Native uniform-grid vorticity and Q-criterion kernels."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from naviertwin._native import _kernels

if _kernels is None:  # pragma: no cover
    raise ImportError("NavierTwin native kernels are required")


def vorticity_2d(
    u: NDArray[np.float64],
    v: NDArray[np.float64],
    dx: float = 1.0,
    dy: float = 1.0,
) -> NDArray[np.float64]:
    """ωz = ∂v/∂x − ∂u/∂y. 배열 shape: (ny, nx)."""
    return np.asarray(
        _kernels.vorticity_2d(np.asarray(u, dtype=np.float64), np.asarray(v, dtype=np.float64), float(dx), float(dy)),
        dtype=np.float64,
    )


def q_criterion_2d(
    u: NDArray[np.float64],
    v: NDArray[np.float64],
    dx: float = 1.0,
    dy: float = 1.0,
) -> NDArray[np.float64]:
    """Q = -½(S²-Ω²) = ½(|Ω|²-|S|²) 2D 근사."""
    return np.asarray(
        _kernels.q_criterion_2d(np.asarray(u, dtype=np.float64), np.asarray(v, dtype=np.float64), float(dx), float(dy)),
        dtype=np.float64,
    )


def vorticity_3d(
    u: NDArray[np.float64],
    v: NDArray[np.float64],
    w: NDArray[np.float64],
    dx: float = 1.0, dy: float = 1.0, dz: float = 1.0,
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """(ωx, ωy, ωz). 배열 shape: (nz, ny, nx)."""
    wx, wy, wz = _kernels.vorticity_3d(
        np.asarray(u, dtype=np.float64),
        np.asarray(v, dtype=np.float64),
        np.asarray(w, dtype=np.float64),
        float(dx),
        float(dy),
        float(dz),
    )
    return np.asarray(wx, dtype=np.float64), np.asarray(wy, dtype=np.float64), np.asarray(wz, dtype=np.float64)


__all__ = ["vorticity_2d", "q_criterion_2d", "vorticity_3d"]
