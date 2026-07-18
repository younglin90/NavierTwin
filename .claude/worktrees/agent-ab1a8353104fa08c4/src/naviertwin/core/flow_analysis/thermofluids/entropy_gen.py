"""엔트로피 생성율 계산 (heat + viscous).

Bejan 의 entropy generation rate:
    s_gen = (k / T²) ∇T · ∇T + (μ / T) Φ

여기서 Φ = 2·(e_ij e_ij) - (2/3)(div u)² — viscous dissipation.

2D 필드에 대한 간단 구현 (균일 격자).

Examples:
    >>> import numpy as np
    >>> from naviertwin.core.flow_analysis.thermofluids.entropy_gen import (
    ...     entropy_generation_2d,
    ... )
    >>> u = np.ones((20, 20))
    >>> v = np.zeros((20, 20))
    >>> T = 300.0 + np.random.default_rng(0).standard_normal((20, 20))
    >>> s = entropy_generation_2d(u, v, T, dx=0.1, dy=0.1, mu=1e-3, k=0.026)
    >>> s.shape
    (20, 20)
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from naviertwin._native import _kernels
from naviertwin.utils.logger import get_logger

if _kernels is None:  # pragma: no cover
    raise ImportError("NavierTwin native kernels are required")

logger = get_logger(__name__)


def entropy_generation_2d(
    u: NDArray[np.float64],
    v: NDArray[np.float64],
    T: NDArray[np.float64],
    dx: float = 1.0,
    dy: float = 1.0,
    mu: float = 1.8e-5,
    k: float = 0.026,
) -> NDArray[np.float64]:
    """2D 엔트로피 생성률 필드 (격자 정렬).

    Args:
        u, v: 속도 성분 (ny, nx).
        T: 온도 (ny, nx) — Kelvin.
        dx, dy: 격자 간격.
        mu: 동점성계수 [Pa·s].
        k: 열전도율 [W/(m·K)].

    Returns:
        s_gen (ny, nx).
    """
    s_gen = np.asarray(
        _kernels.entropy_generation_2d(
            np.asarray(u, dtype=np.float64),
            np.asarray(v, dtype=np.float64),
            np.asarray(T, dtype=np.float64),
            float(dx),
            float(dy),
            float(mu),
            float(k),
        ),
        dtype=np.float64,
    )

    logger.debug(
        "entropy_gen: mean=%.3g",
        float(s_gen.mean()),
    )
    return s_gen


__all__ = ["entropy_generation_2d"]
