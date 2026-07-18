"""보존법칙 사후 보정.

Surrogate/ROM 출력이 질량·운동량·에너지 같은 전역 보존량을 정확히
만족하도록 사후 투영한다. 선형 제약 하에서의 최소 변경 투영:

    minimize ‖u' - u‖² s.t. A u' = b

해: u' = u - Aᵀ (A Aᵀ)^{-1} (A u - b).

Examples:
    >>> import numpy as np
    >>> from naviertwin.core.physics_correction.physics_correction import (
    ...     project_linear_constraint, enforce_mass_conservation,
    ... )
    >>> u = np.array([1.0, 2.0, 3.0])
    >>> A = np.array([[1.0, 1.0, 1.0]])
    >>> b = np.array([5.0])
    >>> u_corr = project_linear_constraint(u, A, b)
    >>> float(u_corr.sum())
    5.0
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from naviertwin._native import _kernels
from naviertwin.utils.logger import get_logger

if _kernels is None:  # pragma: no cover
    raise ImportError("NavierTwin native kernels are required")

logger = get_logger(__name__)


def project_linear_constraint(
    u: NDArray[np.float64],
    A: NDArray[np.float64],
    b: NDArray[np.float64],
) -> NDArray[np.float64]:
    """선형 제약 A u = b 를 만족하도록 u 를 최소-변화 투영한다.

    Args:
        u: (..., n) 또는 (n,).
        A: (m, n) 제약 행렬 (행 독립).
        b: (m,) RHS.

    Returns:
        제약을 만족하는 u'.
    """
    u = np.asarray(u, dtype=np.float64)
    A = np.asarray(A, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64).ravel()
    if A.ndim != 2:
        raise ValueError("A 는 2D 이어야 합니다")
    if A.shape[0] != b.size:
        raise ValueError(f"A 행수({A.shape[0]}) != b 크기({b.size})")
    if u.shape[-1] != A.shape[1]:
        raise ValueError(
            f"u 마지막 축({u.shape[-1]}) != A 열수({A.shape[1]})"
        )

    # (A Aᵀ)⁻¹ (A u - b) 를 배치로 처리
    AAT = A @ A.T                   # (m, m)
    # 잔차: (..., m)
    r = np.tensordot(u, A, axes=([-1], [1])) - b
    rhs = r.reshape(-1, b.size).T
    lam = np.asarray(_kernels.solve_square(AAT.T, rhs), dtype=np.float64).T.reshape(r.shape)
    # 수정: Aᵀ lam → (..., n)
    correction = lam @ A            # (..., n)
    return u - correction


def enforce_mass_conservation(
    rho: NDArray[np.float64],
    cell_volumes: NDArray[np.float64],
    target_mass: float,
) -> NDArray[np.float64]:
    """셀별 밀도 rho 를 total mass = target_mass 로 스케일링.

    Args:
        rho: (n_cells,) 또는 (..., n_cells).
        cell_volumes: (n_cells,) 양수.
        target_mass: 목표 총 질량.

    Returns:
        스케일된 rho. 총 질량이 target_mass 가 되도록.
    """
    rho = np.asarray(rho, dtype=np.float64)
    V = np.asarray(cell_volumes, dtype=np.float64)
    if V.ndim != 1 or rho.shape[-1] != V.size:
        raise ValueError("rho 마지막 축과 cell_volumes 길이가 일치해야 합니다")
    total = np.tensordot(rho, V, axes=([-1], [0]))
    if np.any(total == 0):
        raise ValueError("총 질량이 0 인 스냅샷이 있어 스케일링 불가")
    factor = target_mass / total
    # 마지막 축에 broadcast
    return rho * factor[..., None]


def enforce_divergence_free_1d(u: NDArray[np.float64]) -> NDArray[np.float64]:
    """1D 이산 속도 u 에서 평균 0 으로 이동 (div-free 대리)."""
    u = np.asarray(u, dtype=np.float64)
    return u - u.mean(axis=-1, keepdims=True)


__all__ = [
    "project_linear_constraint",
    "enforce_mass_conservation",
    "enforce_divergence_free_1d",
]
