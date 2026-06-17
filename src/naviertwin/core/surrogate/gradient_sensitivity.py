"""Surrogate 그래디언트 + 민감도 분석 — finite difference / sensitivity index.

학습된 surrogate 모델 (predict 함수)에서 입력 변수에 대한 출력의 민감도를
정량화. 기울기 기반 (FD), Morris elementary effect, 분산 기반 등.

상용 툴 대응:
    - SALib: morris, sobol, fast (Morris+Sobol 위주)
    - SciKit-Learn: permutation_importance
    - 학술: Saltelli et al., "Global Sensitivity Analysis. The Primer", 2008.

Examples:
    >>> import numpy as np
    >>> def f(x): return np.sum(x ** 2, axis=1)
    >>> from naviertwin.core.surrogate.gradient_sensitivity import (
    ...     finite_difference_gradient
    ... )
    >>> g = finite_difference_gradient(f, x=np.array([1.0, 2.0, 3.0]))
    >>> # df/dx = 2x → [2, 4, 6]
    >>> abs(g[1] - 4.0) < 1e-3
    True
"""

from __future__ import annotations

from typing import Callable

import numpy as np
from numpy.typing import NDArray

from naviertwin.utils.logger import get_logger

logger = get_logger(__name__)


def finite_difference_gradient(
    f: Callable[[NDArray[np.float64]], NDArray[np.float64]],
    x: NDArray[np.float64],
    h: float = 1e-5,
) -> NDArray[np.float64]:
    """중앙 차분으로 ∂f/∂x_i 근사.

    Args:
        f: 입력 (N, d) 또는 (d,) → 출력 (N,) 또는 스칼라.
        x: (d,) 평가점.
        h: step size.

    Returns:
        (d,) 그래디언트.

    Raises:
        ValueError: x가 1D 아님 또는 h ≤ 0.
    """
    x = np.asarray(x, dtype=np.float64).ravel()
    if h <= 0:
        raise ValueError(f"h must be > 0, got {h}")

    d = x.size
    grad = np.zeros(d)
    i = 0
    while i < d:
        x_plus = x.copy()
        x_minus = x.copy()
        x_plus[i] += h
        x_minus[i] -= h
        f_plus = f(x_plus[None, :])
        f_minus = f(x_minus[None, :])
        # 스칼라 출력 가정
        f_plus_val = float(np.atleast_1d(f_plus)[0])
        f_minus_val = float(np.atleast_1d(f_minus)[0])
        grad[i] = (f_plus_val - f_minus_val) / (2.0 * h)
        i += 1
    return grad


def morris_elementary_effects(
    f: Callable[[NDArray[np.float64]], NDArray[np.float64]],
    bounds: NDArray[np.float64],
    n_trajectories: int = 10,
    n_levels: int = 4,
    seed: int | None = 0,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Morris elementary effects (Morris 1991) — μ*, σ.

    각 변수에 대해 무작위 방향으로 한 단계 변화시킨 출력 변화량 ee_i.
    μ* = E[|ee|] (전반적 영향), σ = std(ee) (비선형/상호작용).

    Args:
        f: surrogate 함수.
        bounds: (d, 2) 각 변수의 [low, high].
        n_trajectories: 표본 trajectory 수.
        n_levels: grid 분할 수.
        seed: RNG.

    Returns:
        (mu_star, sigma): 둘 다 (d,).

    Raises:
        ValueError: bounds 형상 오류.
    """
    bounds = np.asarray(bounds, dtype=np.float64)
    if bounds.ndim != 2 or bounds.shape[1] != 2:
        raise ValueError(f"bounds must be (d, 2), got {bounds.shape}")
    if n_trajectories <= 0 or n_levels < 2:
        raise ValueError(
            f"n_trajectories > 0 and n_levels >= 2 required, "
            f"got {n_trajectories}, {n_levels}"
        )

    d = bounds.shape[0]
    rng = np.random.default_rng(seed)
    delta = 1.0 / (n_levels - 1)

    ees = np.zeros((n_trajectories, d))
    t = 0
    while t < n_trajectories:
        # 무작위 시작 점 (0, delta, 2*delta, ...)
        x0 = rng.integers(0, n_levels, d) / (n_levels - 1)
        order = rng.permutation(d)
        x_curr = x0.copy()
        order_idx = 0
        while order_idx < order.size:
            i = order[order_idx]
            x_next = x_curr.copy()
            sign = 1.0 if x_curr[i] + delta <= 1.0 else -1.0
            x_next[i] += sign * delta
            # 실제 좌표 매핑
            real_curr = bounds[:, 0] + x_curr * (bounds[:, 1] - bounds[:, 0])
            real_next = bounds[:, 0] + x_next * (bounds[:, 1] - bounds[:, 0])
            f_curr = float(np.atleast_1d(f(real_curr[None, :]))[0])
            f_next = float(np.atleast_1d(f(real_next[None, :]))[0])
            ees[t, i] = (f_next - f_curr) / (sign * delta)
            x_curr = x_next
            order_idx += 1
        t += 1

    mu_star = np.mean(np.abs(ees), axis=0)
    sigma = np.std(ees, axis=0)
    return mu_star, sigma


def variance_decomposition_1d(
    f: Callable[[NDArray[np.float64]], NDArray[np.float64]],
    bounds: NDArray[np.float64],
    n_samples: int = 1000,
    n_levels: int = 10,
    seed: int | None = 0,
) -> NDArray[np.float64]:
    """1차 Sobol-like 인덱스 (단순화 — variance attributable to single var).

    Var(E[Y|X_i]) / Var(Y) 추정. 단순한 conditional binning 방식.

    Args:
        f: surrogate.
        bounds: (d, 2).
        n_samples: 표본 수.
        n_levels: 각 변수 bin 수.
        seed: RNG.

    Returns:
        (d,) S1 추정.

    Raises:
        ValueError: bounds 오류.
    """
    bounds = np.asarray(bounds, dtype=np.float64)
    if bounds.ndim != 2 or bounds.shape[1] != 2:
        raise ValueError(f"bounds must be (d, 2), got {bounds.shape}")

    rng = np.random.default_rng(seed)
    d = bounds.shape[0]
    X = rng.uniform(bounds[:, 0], bounds[:, 1], (n_samples, d))
    Y = np.atleast_1d(f(X)).ravel()
    if Y.shape[0] != n_samples:
        raise ValueError(f"f output shape mismatch: {Y.shape}")

    var_total = float(Y.var()) + 1e-30
    S1 = np.zeros(d)
    i = 0
    while i < d:
        bins = np.linspace(bounds[i, 0], bounds[i, 1], n_levels + 1)
        bin_idx = np.clip(
            np.digitize(X[:, i], bins) - 1, 0, n_levels - 1,
        )
        bin_counts = np.bincount(bin_idx, minlength=n_levels).astype(np.float64)
        bin_sums = np.bincount(bin_idx, weights=Y, minlength=n_levels)
        bin_means = np.divide(
            bin_sums,
            bin_counts,
            out=np.zeros(n_levels, dtype=np.float64),
            where=bin_counts > 0,
        )
        # weighted bin-mean aggregation
        total = bin_counts.sum()
        if total > 0:
            global_mean = (bin_means * bin_counts / total).sum()
            cond_var = (
                bin_counts / total * (bin_means - global_mean) ** 2
            ).sum()
        else:
            cond_var = 0.0
        S1[i] = float(cond_var / var_total)
        i += 1
    return S1


def permutation_importance(
    f: Callable[[NDArray[np.float64]], NDArray[np.float64]],
    X: NDArray[np.float64],
    y: NDArray[np.float64],
    n_repeats: int = 5,
    seed: int | None = 0,
) -> NDArray[np.float64]:
    """Permutation importance — 변수 셔플 후 RMSE 증가량.

    Args:
        f: surrogate (X 받아 ŷ 반환).
        X: (n, d) 검증 데이터.
        y: (n,) 진짜 출력.
        n_repeats: 셔플 반복 횟수.
        seed: RNG.

    Returns:
        (d,) 평균 RMSE 증가량.

    Raises:
        ValueError: 형상 오류.
    """
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64).ravel()
    if X.ndim != 2 or X.shape[0] != y.shape[0]:
        raise ValueError(f"shape error: {X.shape} vs {y.shape}")

    rng = np.random.default_rng(seed)
    y_pred_base = np.atleast_1d(f(X)).ravel()
    rmse_base = float(np.sqrt(np.mean((y - y_pred_base) ** 2)))

    d = X.shape[1]
    importance = np.zeros(d)
    i = 0
    while i < d:
        diffs = np.zeros(n_repeats, dtype=np.float64)
        repeat_idx = 0
        while repeat_idx < n_repeats:
            X_perm = X.copy()
            perm = rng.permutation(X.shape[0])
            X_perm[:, i] = X[perm, i]
            y_perm = np.atleast_1d(f(X_perm)).ravel()
            rmse_perm = float(np.sqrt(np.mean((y - y_perm) ** 2)))
            diffs[repeat_idx] = rmse_perm - rmse_base
            repeat_idx += 1
        importance[i] = float(np.mean(diffs))
        i += 1
    return importance


__all__ = [
    "finite_difference_gradient",
    "morris_elementary_effects",
    "variance_decomposition_1d",
    "permutation_importance",
]
