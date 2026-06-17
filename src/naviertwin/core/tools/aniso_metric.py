"""Anisotropic mesh metric — Hessian-based metric tensor.

M(x) = |H_f(x)| (절대값 고유분해), 길이 측정 ds² = dx M dx.

Examples:
    >>> import numpy as np
    >>> from naviertwin.core.tools.aniso_metric import metric_from_hessian
    >>> H = np.eye(2) * np.array([[2.0]])
    >>> H = np.array([np.eye(2)*2])
    >>> M = metric_from_hessian(H)
    >>> M.shape
    (1, 2, 2)
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from naviertwin._native import _kernels

if _kernels is None:  # pragma: no cover
    raise ImportError("NavierTwin native kernels are required by anisotropic metrics")


def metric_from_hessian(
    H: NDArray[np.float64], *, h_min: float = 1e-3, h_max: float = 1.0,
) -> NDArray[np.float64]:
    """|H| 의 spectrum 을 [1/h_max², 1/h_min²] 에 clip → metric tensor."""
    return _kernels.metric_from_hessian_2d(np.asarray(H, dtype=np.float64), float(h_min), float(h_max))


def edge_length_metric(
    M_a: NDArray[np.float64], M_b: NDArray[np.float64],
    a: NDArray[np.float64], b: NDArray[np.float64],
) -> float:
    """metric-induced edge length (mid-point average metric)."""
    return float(
        _kernels.edge_length_metric(
            np.asarray(M_a, dtype=np.float64),
            np.asarray(M_b, dtype=np.float64),
            np.asarray(a, dtype=np.float64),
            np.asarray(b, dtype=np.float64),
        ),
    )


__all__ = ["edge_length_metric", "metric_from_hessian"]
