"""통계 거리 — Wasserstein, MMD, KL divergence.

생성 모델 품질 평가, surrogate 분포 비교 등에 사용.

Examples:
    >>> import numpy as np
    >>> from naviertwin.core.validation.distances import (
    ...     wasserstein_1d, mmd_gaussian, kl_divergence_gaussian,
    ... )
    >>> rng = np.random.default_rng(0)
    >>> x = rng.standard_normal(500)
    >>> y = rng.standard_normal(500)
    >>> float(wasserstein_1d(x, y)) < 0.2
    True
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from naviertwin.utils.logger import get_logger

logger = get_logger(__name__)


def wasserstein_1d(
    x: NDArray[np.float64], y: NDArray[np.float64]
) -> float:
    """1-Wasserstein 거리 (1D): W₁ = ∫|F_x⁻¹ - F_y⁻¹| du.

    정렬 후 평균 절대 차이로 계산.
    """
    x = np.sort(np.asarray(x, dtype=np.float64))
    y = np.sort(np.asarray(y, dtype=np.float64))
    n = max(len(x), len(y))
    xq = np.interp(np.linspace(0, 1, n), np.linspace(0, 1, len(x)), x)
    yq = np.interp(np.linspace(0, 1, n), np.linspace(0, 1, len(y)), y)
    return float(np.mean(np.abs(xq - yq)))


def mmd_gaussian(
    X: NDArray[np.float64],
    Y: NDArray[np.float64],
    sigma: float = 1.0,
) -> float:
    """Maximum Mean Discrepancy with Gaussian kernel.

    MMD²(X, Y) = E[k(X,X')] + E[k(Y,Y')] - 2 E[k(X,Y)].

    Args:
        X: (N, d).
        Y: (M, d).
        sigma: RBF 커널 폭.
    """
    X = np.asarray(X, dtype=np.float64)
    Y = np.asarray(Y, dtype=np.float64)
    if X.ndim == 1:
        X = X[:, None]
    if Y.ndim == 1:
        Y = Y[:, None]
    if X.shape[1] != Y.shape[1]:
        raise ValueError("X, Y 차원 불일치")

    def rbf_kernel_sum(A: NDArray[np.float64], B: NDArray[np.float64]) -> float:
        sq = np.sum(A ** 2, axis=1)[:, None] + np.sum(B ** 2, axis=1)[None, :] - 2 * A @ B.T
        return float(np.exp(-sq / (2 * sigma ** 2)).mean())

    return float(max(rbf_kernel_sum(X, X) + rbf_kernel_sum(Y, Y) - 2 * rbf_kernel_sum(X, Y), 0.0))


def kl_divergence_gaussian(
    mu1: NDArray[np.float64],
    cov1: NDArray[np.float64],
    mu2: NDArray[np.float64],
    cov2: NDArray[np.float64],
) -> float:
    """두 다변량 정규분포 간 KL divergence D(N_1 || N_2).

        D = 0.5 · [ tr(Σ_2⁻¹ Σ_1) + (μ_2-μ_1)ᵀ Σ_2⁻¹ (μ_2-μ_1)
                   - k + ln(|Σ_2| / |Σ_1|) ]
    """
    mu1 = np.asarray(mu1, dtype=np.float64).ravel()
    mu2 = np.asarray(mu2, dtype=np.float64).ravel()
    cov1 = np.asarray(cov1, dtype=np.float64)
    cov2 = np.asarray(cov2, dtype=np.float64)
    k = mu1.size
    cov2_inv = np.linalg.inv(cov2)
    diff = mu2 - mu1
    term_trace = float(np.trace(cov2_inv @ cov1))
    term_quad = float(diff @ cov2_inv @ diff)
    sign1, logdet1 = np.linalg.slogdet(cov1)
    sign2, logdet2 = np.linalg.slogdet(cov2)
    if sign1 <= 0 or sign2 <= 0:
        raise ValueError("공분산 행렬이 positive definite 이어야 합니다")
    return 0.5 * (term_trace + term_quad - k + (logdet2 - logdet1))


__all__ = ["wasserstein_1d", "mmd_gaussian", "kl_divergence_gaussian"]
