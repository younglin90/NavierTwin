"""Quadrature on cut cells — Monte-Carlo with rejection (간단).

Examples:
    >>> import numpy as np
    >>> from naviertwin.core.geometry.cut_quadrature import mc_integrate
    >>> rng = np.random.default_rng(0)
    >>> # disk area ≈ π·0.25 ≈ 0.785
    >>> area = mc_integrate(
    ...     lambda p: float(p[0]**2 + p[1]**2 < 1.0),
    ...     bbox=((-1, -1), (1, 1)), n_samples=20000, rng=rng,
    ... )
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np


def mc_integrate(
    f: Callable[[np.ndarray], float],
    *,
    bbox: tuple[tuple[float, float], tuple[float, float]],
    n_samples: int = 5000,
    rng: np.random.Generator | None = None,
) -> float:
    rng = rng if rng is not None else np.random.default_rng(0)
    (x0, y0), (x1, y1) = bbox
    pts = np.column_stack([
        rng.uniform(x0, x1, n_samples),
        rng.uniform(y0, y1, n_samples),
    ])
    vals = np.fromiter(map(f, pts), dtype=float, count=n_samples)
    area = (x1 - x0) * (y1 - y0)
    return float(vals.mean()) * area


__all__ = ["mc_integrate"]
