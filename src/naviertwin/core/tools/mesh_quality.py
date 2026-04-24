"""삼각형 메쉬 품질 지표 — aspect ratio, skewness, min-angle.

Examples:
    >>> import numpy as np
    >>> from naviertwin.core.tools.mesh_quality import aspect_ratio_triangle
    >>> tri = np.array([[0,0],[1,0],[0.5, np.sqrt(3)/2]])
    >>> abs(aspect_ratio_triangle(tri) - 1.0) < 1e-9
    True
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def aspect_ratio_triangle(tri: NDArray[np.float64]) -> float:
    """AR = R / (2r) (R circumradius, r inradius). 정삼각 = 1."""
    A, B, C = tri
    a = np.linalg.norm(B - C)
    b = np.linalg.norm(A - C)
    c = np.linalg.norm(A - B)
    s = 0.5 * (a + b + c)
    area = max(np.sqrt(max(s * (s - a) * (s - b) * (s - c), 1e-30)), 1e-30)
    R = (a * b * c) / (4.0 * area)
    r = area / s
    return float(R / (2.0 * r))


def min_angle_deg(tri: NDArray[np.float64]) -> float:
    A, B, C = tri
    def _ang(p, q, r):
        u = q - p
        v = r - p
        cos = (u @ v) / (np.linalg.norm(u) * np.linalg.norm(v) + 1e-30)
        return np.degrees(np.arccos(np.clip(cos, -1, 1)))
    return float(min(_ang(A, B, C), _ang(B, A, C), _ang(C, A, B)))


def skewness_equilateral(tri: NDArray[np.float64]) -> float:
    """skewness = (60 - θ_min) / 60."""
    theta_min = min_angle_deg(tri)
    return float(max(0.0, (60.0 - theta_min) / 60.0))


def mesh_quality_report(
    points: NDArray[np.float64], simplices: NDArray[np.int64],
) -> dict[str, float]:
    ar = [aspect_ratio_triangle(points[t]) for t in simplices]
    sk = [skewness_equilateral(points[t]) for t in simplices]
    an = [min_angle_deg(points[t]) for t in simplices]
    return {
        "mean_aspect": float(np.mean(ar)),
        "max_aspect": float(np.max(ar)),
        "mean_skew": float(np.mean(sk)),
        "max_skew": float(np.max(sk)),
        "min_angle_deg": float(np.min(an)),
    }


__all__ = [
    "aspect_ratio_triangle", "min_angle_deg", "skewness_equilateral",
    "mesh_quality_report",
]
