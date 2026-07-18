"""파라미터 스윕 — LHS / Sobol / Halton / grid / random.

scipy.stats.qmc 를 활용한 저차원 저불일치 샘플링. 실패 시 random fallback.

Examples:
    >>> from naviertwin.core.sampling.param_sweep import generate_sweep
    >>> pts = generate_sweep([(0.1, 1.0), (100, 500)], n_points=16, kind="lhs")
    >>> pts.shape
    (16, 2)
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
from numpy.typing import NDArray

from naviertwin._native import _kernels
from naviertwin.utils.logger import get_logger

if _kernels is None:  # pragma: no cover
    raise ImportError("NavierTwin native kernels are required")

logger = get_logger(__name__)


def _scale(unit: NDArray[np.float64], bounds: Sequence[tuple[float, float]]) -> NDArray[np.float64]:
    return _kernels.scale_to_bounds(
        np.asarray(unit, dtype=np.float64),
        np.asarray(bounds, dtype=np.float64),
    )


def generate_sweep(
    bounds: Sequence[tuple[float, float]],
    n_points: int,
    kind: str = "lhs",
    seed: int | None = 0,
) -> NDArray[np.float64]:
    """파라미터 샘플링.

    Args:
        bounds: [(low, high), ...] 각 차원 범위.
        n_points: 생성 포인트 수.
        kind: "lhs" / "sobol" / "halton" / "random" / "grid".
        seed: 재현.

    Returns:
        (n_points, dim) 실수 행렬.
    """
    dim = len(bounds)
    if kind == "grid":
        per = max(1, int(round(n_points ** (1 / dim))))
        pts = _kernels.regular_grid_points(np.asarray(bounds, dtype=np.float64), per)
        if pts.shape[0] > n_points:
            rng = np.random.default_rng(seed)
            idx = rng.choice(pts.shape[0], size=n_points, replace=False)
            pts = pts[idx]
        return pts

    try:
        from scipy.stats import qmc
    except ImportError:
        logger.warning("scipy.stats.qmc 없음 → random")
        kind = "random"

    if kind == "random":
        rng = np.random.default_rng(seed)
        unit = rng.random((n_points, dim))
    elif kind == "lhs":
        sampler = qmc.LatinHypercube(d=dim, seed=seed)
        unit = sampler.random(n=n_points)
    elif kind == "sobol":
        sampler = qmc.Sobol(d=dim, seed=seed, scramble=True)
        unit = sampler.random(n=n_points)
    elif kind == "halton":
        sampler = qmc.Halton(d=dim, seed=seed, scramble=True)
        unit = sampler.random(n=n_points)
    else:
        raise ValueError(f"알 수 없는 kind: {kind}")

    pts = _scale(unit, bounds)
    logger.info("sweep(%s): %d points × %d dims", kind, n_points, dim)
    return pts


__all__ = ["generate_sweep"]
