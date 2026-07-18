"""Monotone convergence test — successive errors strictly decreasing.

Examples:
    >>> from naviertwin.core.verification.monotone import is_monotone_decreasing
    >>> is_monotone_decreasing([1.0, 0.5, 0.25])
    True
"""

from __future__ import annotations

from collections.abc import Sequence

from naviertwin._native import _kernels

if _kernels is None:  # pragma: no cover
    raise ImportError("NavierTwin native kernels are required")


def is_monotone_decreasing(errs: Sequence[float], *, atol: float = 0.0) -> bool:
    return bool(_kernels.is_monotone_decreasing(list(errs), float(atol)))


def convergence_ratio(errs: Sequence[float]) -> list[float]:
    return list(_kernels.convergence_ratio(list(errs)))


__all__ = ["convergence_ratio", "is_monotone_decreasing"]
