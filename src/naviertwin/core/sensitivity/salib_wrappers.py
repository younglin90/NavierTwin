"""SALib 전체 민감도 분석 래퍼 — Morris, FAST, PAWN, Delta.

기존 `sobol_analysis.py` 는 Sobol 만. 이 모듈은 SALib 의 다양한 방법을 편하게.

Examples:
    >>> import numpy as np
    >>> from naviertwin.core.sensitivity.salib_wrappers import (
    ...     morris_analysis, fast_analysis,
    ... )
    >>> def model(X):
    ...     return np.sin(X[:, 0]) + 0.5 * X[:, 1] ** 2
    >>> problem = {
    ...     "num_vars": 2,
    ...     "names": ["x1", "x2"],
    ...     "bounds": [[-np.pi, np.pi], [-1, 1]],
    ... }
    >>> mu, sigma = morris_analysis(problem, model, n_trajectories=20)
    >>> mu.shape == (2,)
    True
"""

from __future__ import annotations

from typing import Callable

import numpy as np
from numpy.typing import NDArray

from naviertwin.utils.logger import get_logger

logger = get_logger(__name__)


def _require_salib() -> None:
    try:
        import SALib  # noqa: F401
    except ImportError as exc:
        raise RuntimeError("SALib 필요: pip install SALib") from exc


def morris_analysis(
    problem: dict,
    model: Callable[[NDArray[np.float64]], NDArray[np.float64]],
    n_trajectories: int = 50,
    seed: int | None = 0,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Morris elementary effects 분석.

    Returns:
        (mu_star, sigma) — 각 입력의 영향도/상호작용성.
    """
    _require_salib()
    from SALib.analyze import morris as ana_morris
    from SALib.sample import morris as samp_morris

    X = samp_morris.sample(problem, N=n_trajectories, num_levels=4, seed=seed)
    Y = np.asarray(model(X), dtype=np.float64).ravel()
    res = ana_morris.analyze(
        problem, X, Y, conf_level=0.95, print_to_console=False, num_levels=4,
    )
    mu_star = np.asarray(res["mu_star"], dtype=np.float64)
    sigma = np.asarray(res["sigma"], dtype=np.float64)
    logger.info("Morris: μ*=%s, σ=%s", mu_star.round(4).tolist(), sigma.round(4).tolist())
    return mu_star, sigma


def fast_analysis(
    problem: dict,
    model: Callable[[NDArray[np.float64]], NDArray[np.float64]],
    n_samples: int = 1024,
    seed: int | None = 0,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Fourier Amplitude Sensitivity Test (FAST) — S1, ST.

    Returns:
        (S1, ST).
    """
    _require_salib()
    from SALib.analyze import fast as ana_fast
    from SALib.sample import fast_sampler

    X = fast_sampler.sample(problem, N=n_samples, seed=seed)
    Y = np.asarray(model(X), dtype=np.float64).ravel()
    res = ana_fast.analyze(problem, Y, print_to_console=False)
    S1 = np.asarray(res["S1"], dtype=np.float64)
    ST = np.asarray(res["ST"], dtype=np.float64)
    logger.info("FAST: S1=%s, ST=%s", S1.round(4).tolist(), ST.round(4).tolist())
    return S1, ST


def pawn_analysis(
    problem: dict,
    model: Callable[[NDArray[np.float64]], NDArray[np.float64]],
    n_samples: int = 1024,
    S: int = 10,
    seed: int | None = 0,
) -> NDArray[np.float64]:
    """PAWN density-based sensitivity — 중앙값 KS 통계.

    Returns:
        각 변수의 median PAWN index.
    """
    _require_salib()
    from SALib.analyze import pawn
    from SALib.sample import latin

    X = latin.sample(problem, N=n_samples, seed=seed)
    Y = np.asarray(model(X), dtype=np.float64).ravel()
    res = pawn.analyze(problem, X, Y, S=S, print_to_console=False)
    median = np.asarray(res["median"], dtype=np.float64)
    logger.info("PAWN median=%s", median.round(4).tolist())
    return median


def delta_analysis(
    problem: dict,
    model: Callable[[NDArray[np.float64]], NDArray[np.float64]],
    n_samples: int = 1024,
    seed: int | None = 0,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Borgonovo δ moment-independent importance + S1.

    Returns:
        (delta, S1).
    """
    _require_salib()
    from SALib.analyze import delta
    from SALib.sample import latin

    X = latin.sample(problem, N=n_samples, seed=seed)
    Y = np.asarray(model(X), dtype=np.float64).ravel()
    res = delta.analyze(problem, X, Y, print_to_console=False)
    d = np.asarray(res["delta"], dtype=np.float64)
    s1 = np.asarray(res["S1"], dtype=np.float64)
    logger.info("Delta: δ=%s, S1=%s", d.round(4).tolist(), s1.round(4).tolist())
    return d, s1


__all__ = [
    "morris_analysis",
    "fast_analysis",
    "pawn_analysis",
    "delta_analysis",
]
