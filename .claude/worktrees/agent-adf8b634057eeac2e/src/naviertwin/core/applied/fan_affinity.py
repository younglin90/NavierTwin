"""Fan affinity laws — Q ∝ N, H ∝ N², P ∝ N³.

Examples:
    >>> from naviertwin.core.applied.fan_affinity import scale_Q_H_P
    >>> scale_Q_H_P(Q1=10, H1=20, P1=300, N1=1000, N2=1500)
    (15.0, 45.0, 1012.5)
"""

from __future__ import annotations


def scale_Q_H_P(*, Q1: float, H1: float, P1: float,
                  N1: float, N2: float) -> tuple[float, float, float]:
    r = N2 / N1
    return float(Q1 * r), float(H1 * r * r), float(P1 * r * r * r)


__all__ = ["scale_Q_H_P"]
