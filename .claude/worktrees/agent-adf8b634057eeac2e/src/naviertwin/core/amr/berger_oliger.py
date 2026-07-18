"""Berger-Oliger AMR — recursive subcycling time scheduler.

각 레벨 ℓ 가 dt_ℓ = dt_0 / r^ℓ 로 r 번 substep.

Examples:
    >>> from naviertwin.core.amr.berger_oliger import schedule
    >>> calls = schedule(level=0, max_level=2, refine_ratio=2)
    >>> len(calls)
    7
"""

from __future__ import annotations

from naviertwin._native import _kernels


def schedule(
    *, level: int = 0, max_level: int = 2, refine_ratio: int = 2,
) -> list[int]:
    """nested sub-cycling 호출 순서 (각 entry = 호출되는 레벨)."""
    if _kernels is None:
        raise ImportError("NavierTwin native kernels are required by schedule")
    return list(_kernels.schedule_berger_oliger(level, max_level, refine_ratio))


__all__ = ["schedule"]
