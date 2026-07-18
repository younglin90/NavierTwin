"""Curriculum scheduler — easy → hard sampling.

Examples:
    >>> from naviertwin.utils.curriculum import linear_difficulty
    >>> linear_difficulty(epoch=5, max_epoch=10, d_min=0.1, d_max=1.0)
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def linear_difficulty(
    *, epoch: int, max_epoch: int, d_min: float = 0.1, d_max: float = 1.0,
) -> float:
    t = min(1.0, epoch / max(max_epoch, 1))
    return float(d_min + (d_max - d_min) * t)


def select_curriculum_indices(
    difficulties: NDArray[np.float64], *, threshold: float,
) -> NDArray[np.int_]:
    """Return indices with difficulty <= threshold."""
    d = np.asarray(difficulties)
    return np.where(d <= threshold)[0]


__all__ = ["linear_difficulty", "select_curriculum_indices"]
