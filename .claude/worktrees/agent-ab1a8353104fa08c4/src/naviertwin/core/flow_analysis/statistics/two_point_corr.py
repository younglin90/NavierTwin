"""두점 공간 상관 (Two-point correlation) 모듈.

균일 격자에서 한 방향의 공간 상관 계수와 적분 길이 스케일을 계산한다.
난류 특성 평가에 쓰인다.

    R(r) = <u'(x) · u'(x+r)> / <u'(x)²>
    L_int = ∫_0^∞ R(r) dr   (첫 영점까지의 면적)

Examples:
    >>> import numpy as np
    >>> from naviertwin.core.flow_analysis.statistics.two_point_corr import (
    ...     two_point_correlation, integral_length_scale,
    ... )
    >>> rng = np.random.default_rng(0)
    >>> u = rng.standard_normal((1000, 64))  # (n_samples, n_space)
    >>> R = two_point_correlation(u, axis=1)
    >>> L = integral_length_scale(R, dx=0.1)
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from naviertwin.utils.logger import get_logger

logger = get_logger(__name__)


def two_point_correlation(
    u: NDArray[np.float64], axis: int = -1
) -> NDArray[np.float64]:
    """두점 정규화 상관 R(r).

    Args:
        u: 샘플 집합. 마지막 축(기본)이 공간 축이어야 한다.
        axis: 공간 축 인덱스.

    Returns:
        R: shape = (n_space,) — r=0,1,...,n/2 분리에 대한 정규화 상관.
    """
    u = np.asarray(u, dtype=np.float64)
    u = np.moveaxis(u, axis, -1)
    u_prime = u - u.mean(axis=-1, keepdims=True)
    n_space = u_prime.shape[-1]

    # FFT 기반 자기상관 (선형 상관 추정을 위해 zero-padding)
    U = np.fft.rfft(u_prime, n=2 * n_space, axis=-1)
    acf_full = np.fft.irfft(U * np.conj(U), n=2 * n_space, axis=-1)
    # 양의 lag 만
    acf = acf_full[..., :n_space]
    # lag 별 유효 표본 수 정규화
    norm = (n_space - np.arange(n_space)).astype(np.float64)
    acf = acf / norm
    # 앙상블 평균 (마지막 축 제외 모든 축)
    reduce_axes = tuple(range(acf.ndim - 1))
    acf_mean = acf.mean(axis=reduce_axes) if reduce_axes else acf
    R0 = acf_mean[0] if acf_mean[0] != 0 else 1.0
    R = acf_mean / R0
    return R.astype(np.float64)


def integral_length_scale(
    R: NDArray[np.float64], dx: float = 1.0
) -> float:
    """R(r) 의 첫 번째 영점까지 적분하여 적분 길이 스케일을 반환한다.

    Args:
        R: two_point_correlation 결과.
        dx: 격자 간격.

    Returns:
        L_int [m].
    """
    R = np.asarray(R, dtype=np.float64)
    # 첫 영점 찾기
    zero_cross = np.where(R[:-1] * R[1:] <= 0)[0]
    end = int(zero_cross[0]) + 1 if len(zero_cross) else len(R)
    L = float(np.trapezoid(R[:end], dx=dx))
    logger.debug("integral_length_scale: end=%d, L=%.4g", end, L)
    return L


__all__ = ["two_point_correlation", "integral_length_scale"]
