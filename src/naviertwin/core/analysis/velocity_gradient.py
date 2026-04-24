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


def decompose_J_3x3(
    J: NDArray[np.float64],
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """J = S + Ω, S = (J + Jᵀ)/2, Ω = (J - Jᵀ)/2."""
    J = np.asarray(J, dtype=np.float64)
    if J.shape != (3, 3):
        raise ValueError("3x3 expected")
    S = 0.5 * (J + J.T)
    W = 0.5 * (J - J.T)
    return S, W


def invariants_3x3(J: NDArray[np.float64]) -> dict[str, float]:
    """속도장 gradient 불변량 — P, Q, R (nondiv flow 이면 P=0).

    P = -tr J, Q = ½(P² - tr(J²)), R = -det J.
    """
    J = np.asarray(J, dtype=np.float64)
    P = -float(np.trace(J))
    Q = 0.5 * (P ** 2 - float(np.trace(J @ J)))
    R = -float(np.linalg.det(J))
    return {"P": P, "Q": Q, "R": R}


def field_J_2d(
    u: NDArray[np.float64], v: NDArray[np.float64],
    dx: float = 1.0, dy: float = 1.0,
) -> NDArray[np.float64]:
    """격자의 2D velocity gradient J (ny, nx, 2, 2)."""
    du_dx = np.gradient(u, dx, axis=1)
    du_dy = np.gradient(u, dy, axis=0)
    dv_dx = np.gradient(v, dx, axis=1)
    dv_dy = np.gradient(v, dy, axis=0)
    J = np.stack([
        np.stack([du_dx, du_dy], axis=-1),
        np.stack([dv_dx, dv_dy], axis=-1),
    ], axis=-2)
    return J


__all__ = ["decompose_J_3x3", "invariants_3x3", "field_J_2d"]
