"""1D 적응 격자 세분화 — 오차 지표 기반.

Examples:
    >>> import numpy as np
    >>> from naviertwin.core.tools.mesh_refine_1d import refine_by_gradient
    >>> x = np.linspace(0, 1, 11)
    >>> f = np.where(x < 0.5, 0.0, 1.0)
    >>> x2, f2 = refine_by_gradient(x, f, threshold=0.1)
    >>> len(x2) > len(x)
    True
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from naviertwin._native import _kernels

if _kernels is None:  # pragma: no cover
    raise ImportError("NavierTwin native kernels are required")


def refine_by_gradient(
    x: NDArray[np.float64], f: NDArray[np.float64],
    *, threshold: float = 0.1, max_passes: int = 5,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """셀별 |Δf| > threshold 면 중간점 삽입 (선형 보간)."""
    return _kernels.mesh_refine_by_gradient(
        np.asarray(x, dtype=np.float64),
        np.asarray(f, dtype=np.float64),
        float(threshold),
        int(max_passes),
    )


def coarsen_by_tolerance(
    x: NDArray[np.float64], f: NDArray[np.float64],
    *, tol: float = 1e-3,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """인접 두 셀의 2차 차분이 작으면 중간점 제거."""
    return _kernels.mesh_coarsen_by_tolerance(
        np.asarray(x, dtype=np.float64),
        np.asarray(f, dtype=np.float64),
        float(tol),
    )


__all__ = ["refine_by_gradient", "coarsen_by_tolerance"]
