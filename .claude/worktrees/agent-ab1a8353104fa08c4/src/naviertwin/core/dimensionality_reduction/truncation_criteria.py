"""POD/SVD 절단 차수 자동 선택 기준.

ROM/AI 모델에서 몇 개의 모드를 유지할지를 자동 결정. 누적 에너지,
Eckart-Young 노름 오차, scree elbow detection, AIC/BIC.

References:
    Eckart, C. & Young, G., "The approximation of one matrix by another of
    lower rank", Psychometrika, 1936.
    Cattell, R.B., "The Scree Test: Factor Count Selection", 1966.

Examples:
    >>> import numpy as np
    >>> from numpy.linalg import svd as _svd
    >>> rng = np.random.default_rng(0)
    >>> X = rng.standard_normal((50, 30))
    >>> _, s, _ = _svd(X, full_matrices=False)
    >>> from naviertwin.core.dimensionality_reduction.truncation_criteria import (
    ...     truncate_by_energy
    ... )
    >>> r = truncate_by_energy(s, fraction=0.99)
    >>> 1 <= r <= len(s)
    True
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from naviertwin.utils.logger import get_logger

logger = get_logger(__name__)


def truncate_by_energy(
    singular_values: NDArray[np.float64],
    fraction: float = 0.99,
) -> int:
    """누적 에너지 분율이 fraction 이상인 최소 모드 수 r.

    Energy_r = Σ_{i=1}^r σ_i² / Σ_{i=1}^N σ_i².

    Args:
        singular_values: (N,) 정렬된 특이값 (내림차순).
        fraction: 0 < f ≤ 1, 보존할 에너지 비율.

    Returns:
        최소 r.

    Raises:
        ValueError: fraction 범위 오류 또는 빈 입력.
    """
    s = np.asarray(singular_values, dtype=np.float64).ravel()
    if len(s) == 0:
        raise ValueError("singular_values is empty")
    if not (0 < fraction <= 1):
        raise ValueError(f"fraction must be in (0, 1], got {fraction}")

    energies = s ** 2
    total = energies.sum()
    if total < 1e-30:
        return 1
    cum = np.cumsum(energies) / total
    r = int(np.searchsorted(cum, fraction) + 1)
    return min(r, len(s))


def truncate_by_eckart_young(
    singular_values: NDArray[np.float64],
    rel_error: float = 0.01,
) -> int:
    """Eckart-Young: rank-r 절단의 Frobenius 상대 오차가 rel_error 이하.

    ‖A - A_r‖_F² / ‖A‖_F² = (Σ_{i>r} σ_i²) / (Σ σ_i²) ≤ rel_error².

    Args:
        singular_values: 특이값.
        rel_error: 허용 상대 오차.

    Returns:
        최소 r.

    Raises:
        ValueError: rel_error 오류.
    """
    s = np.asarray(singular_values, dtype=np.float64).ravel()
    if len(s) == 0:
        raise ValueError("singular_values is empty")
    if rel_error <= 0:
        raise ValueError(f"rel_error must be > 0, got {rel_error}")

    target = 1.0 - rel_error ** 2
    return truncate_by_energy(s, fraction=target)


def scree_elbow(
    singular_values: NDArray[np.float64],
) -> int:
    """Scree plot의 엘보우 (knee point) 찾기 — 가장 큰 곡률.

    log(σ_i)의 1차 차분이 가장 크게 줄어드는 지점.

    Args:
        singular_values: 특이값.

    Returns:
        엘보우 위치 (1-indexed 모드 수).

    Raises:
        ValueError: 입력 길이 부족.
    """
    s = np.asarray(singular_values, dtype=np.float64).ravel()
    if len(s) < 3:
        raise ValueError(f"need at least 3 singular values, got {len(s)}")
    s_pos = s[s > 1e-30]
    if len(s_pos) < 3:
        return 1
    log_s = np.log(s_pos)
    # second difference (curvature)
    second = np.diff(log_s, n=2)
    elbow = int(np.argmin(second)) + 1  # 인덱스 +1 (모드 수)
    return max(1, elbow)


def truncate_by_aic(
    singular_values: NDArray[np.float64],
    n_samples: int,
) -> int:
    """AIC 기준 최적 모드 수 r — 잔차 분산 + 2k 패널티.

    AIC(r) = n log(σ²_residual) + 2 r, where σ²_r = Σ_{i>r} s_i² / n.

    Args:
        singular_values: 특이값.
        n_samples: 표본 크기.

    Returns:
        AIC 최소화 r.

    Raises:
        ValueError: n_samples ≤ 0.
    """
    s = np.asarray(singular_values, dtype=np.float64).ravel()
    if len(s) == 0:
        raise ValueError("singular_values is empty")
    if n_samples <= 0:
        raise ValueError(f"n_samples must be > 0, got {n_samples}")

    n = len(s)
    energies = s ** 2
    rev_cum = np.flip(np.cumsum(np.flip(energies)))  # rev_cum[r] = Σ_{i>=r} s_i²
    aic_values = np.empty(n, dtype=np.float64)
    ranks = np.arange(1, n, dtype=np.float64)
    if n > 1:
        residual = rev_cum[1:] / n_samples
        aic_head = aic_values[:-1]
        aic_head[:] = 2.0 * ranks
        valid = residual >= 1e-30
        aic_head[valid] = n_samples * np.log(residual[valid]) + 2.0 * ranks[valid]
    aic_values[-1] = 2.0 * n  # all modes — zero residual
    r_best = int(np.argmin(aic_values)) + 1
    return r_best


def truncate_by_bic(
    singular_values: NDArray[np.float64],
    n_samples: int,
) -> int:
    """BIC 기준 — 2 패널티를 ln(n)로 교체. 더 보수적 (더 적은 모드).

    Args:
        singular_values: 특이값.
        n_samples: 표본 크기.

    Returns:
        BIC 최소 r.
    """
    s = np.asarray(singular_values, dtype=np.float64).ravel()
    if n_samples <= 0:
        raise ValueError(f"n_samples must be > 0, got {n_samples}")
    n = len(s)
    energies = s ** 2
    rev_cum = np.flip(np.cumsum(np.flip(energies)))
    log_n = np.log(n_samples)
    bic_values = np.empty(n, dtype=np.float64)
    ranks = np.arange(1, n, dtype=np.float64)
    if n > 1:
        residual = rev_cum[1:] / n_samples
        bic_head = bic_values[:-1]
        bic_head[:] = log_n * ranks
        valid = residual >= 1e-30
        bic_head[valid] = n_samples * np.log(residual[valid]) + log_n * ranks[valid]
    bic_values[-1] = log_n * n
    r_best = int(np.argmin(bic_values)) + 1
    return r_best


def relative_l2_error(
    singular_values: NDArray[np.float64],
    r: int,
) -> float:
    """Eckart-Young 상대 L2 오차: ‖A-A_r‖_F / ‖A‖_F.

    Args:
        singular_values: 특이값.
        r: 절단 모드 수.

    Returns:
        상대 오차 ∈ [0, 1].

    Raises:
        ValueError: r 범위 오류.
    """
    s = np.asarray(singular_values, dtype=np.float64).ravel()
    if r < 0 or r > len(s):
        raise ValueError(f"r must be in [0, {len(s)}], got {r}")
    energies = s ** 2
    total = float(energies.sum())
    if total < 1e-30:
        return 0.0
    residual = float(energies[r:].sum())
    return float(np.sqrt(residual / total))


def cumulative_energy_curve(
    singular_values: NDArray[np.float64],
) -> NDArray[np.float64]:
    """누적 에너지 분율 ∈ [0, 1] 곡선.

    Args:
        singular_values: 특이값.

    Returns:
        (N,) cum[r-1] = Σ_{i=1}^r σ_i² / Σ σ_i².
    """
    s = np.asarray(singular_values, dtype=np.float64).ravel()
    energies = s ** 2
    total = energies.sum()
    if total < 1e-30:
        return np.zeros_like(s)
    return np.cumsum(energies) / total


__all__ = [
    "truncate_by_energy",
    "truncate_by_eckart_young",
    "scree_elbow",
    "truncate_by_aic",
    "truncate_by_bic",
    "relative_l2_error",
    "cumulative_energy_curve",
]
