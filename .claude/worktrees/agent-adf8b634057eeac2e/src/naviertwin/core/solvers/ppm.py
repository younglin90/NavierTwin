"""PPM — Colella & Woodward 1984, parabolic reconstruction.

Examples:
    >>> import numpy as np
    >>> from naviertwin.core.solvers.ppm import ppm_face_values
    >>> u = np.array([1., 2, 3, 4, 5])
    >>> uL, uR = ppm_face_values(u)
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from naviertwin._native import _kernels

if _kernels is None:  # pragma: no cover
    raise ImportError("NavierTwin native kernels are required by PPM reconstruction")


def ppm_face_values(u: NDArray[np.float64]) -> tuple[float, float]:
    """5-point stencil face reconstruction at cell i=2."""
    u_face_left, u_face_right = _kernels.ppm_face_values(np.asarray(u, dtype=np.float64))
    return float(u_face_left), float(u_face_right)


def ppm_monotonize(u_im: float, u_i: float, u_ip: float,
                    uL: float, uR: float) -> tuple[float, float]:
    """Colella-Woodward monotonization."""
    left, right = _kernels.ppm_monotonize(
        float(u_im),
        float(u_i),
        float(u_ip),
        float(uL),
        float(uR),
    )
    return float(left), float(right)


__all__ = ["ppm_face_values", "ppm_monotonize"]
