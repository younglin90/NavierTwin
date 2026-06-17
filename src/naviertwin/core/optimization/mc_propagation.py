"""Monte Carlo 불확실성 전파.

입력 분포 p(x) → 모델 f → 출력 분포 p(y).
통계: 평균, 표준편차, 백분위수, CDF 추정.

Examples:
    >>> import numpy as np
    >>> from naviertwin.core.optimization.mc_propagation import propagate_mc
    >>> rng = np.random.default_rng(0)
    >>> def model(x):
    ...     return np.sin(x[:, 0]) + x[:, 1] ** 2
    >>> X = rng.standard_normal((5000, 2))
    >>> stats = propagate_mc(model, X)
    >>> "mean" in stats and "std" in stats
    True
"""

from __future__ import annotations

from typing import Callable

import numpy as np
from numpy.typing import NDArray

from naviertwin._native import _kernels
from naviertwin.utils.logger import get_logger

logger = get_logger(__name__)

if _kernels is None:  # pragma: no cover
    raise ImportError("NavierTwin native kernels are required by MC propagation")


def propagate_mc(
    model: Callable[[NDArray[np.float64]], NDArray[np.float64]],
    X: NDArray[np.float64],
    percentiles: tuple[float, ...] = (5.0, 50.0, 95.0),
) -> dict[str, NDArray[np.float64] | float]:
    """모델 출력에 대한 MC 통계를 계산한다.

    Args:
        model: (N, d) → (N,) 또는 (N, k) 벡터 함수.
        X: 입력 샘플 (N, d).
        percentiles: 보고할 백분위수 튜플.

    Returns:
        dict: {mean, std, percentiles(dict), samples(원본)}.
    """
    X = np.asarray(X, dtype=np.float64)
    if X.ndim != 2:
        raise ValueError(f"X shape={X.shape} (N, d) 2D 필요")

    Y = np.asarray(model(X), dtype=np.float64)
    if Y.ndim == 1:
        Y = Y[:, None]

    mean, std = _kernels.mean_std_axis0(Y)
    mean = np.asarray(mean, dtype=np.float64)
    std = np.asarray(std, dtype=np.float64)
    pct: dict[float, NDArray[np.float64]] = dict(
        map(lambda p: (float(p), np.percentile(Y, p, axis=0)), percentiles),
    )
    logger.info(
        "MC propagation: N=%d, d=%d, out_dim=%d, mean=%s",
        X.shape[0],
        X.shape[1],
        Y.shape[1],
        mean.tolist() if mean.size <= 4 else "...",
    )
    return {
        "mean": mean,
        "std": std,
        "percentiles": pct,
        "samples": Y,
    }


__all__ = ["propagate_mc"]
