"""4D-Var 데이터 동화 — 선형 동역학 가정 간이 구현.

시공간 전체 관측 {y_k} 에 대해 초기 상태 x_0 의 cost 를 최소화:

    J(x_0) = (x_0 - x_b)ᵀ B^{-1} (x_0 - x_b)
           + Σ_k (y_k - H·M^k x_0)ᵀ R^{-1} (y_k - H·M^k x_0)

여기서 M 은 선형 시간 발전자 (x_{k+1} = M x_k).
비선형 M 인 경우 incremental 4D-Var 가 표준이지만, 이 모듈은 MVP.

Examples:
    >>> import numpy as np
    >>> from naviertwin.core.data_assimilation.four_dvar import four_dvar_linear
    >>> rng = np.random.default_rng(0)
    >>> M = np.array([[0.9, 0.1], [0.0, 0.95]])
    >>> H = np.eye(2)
    >>> x_true = np.array([1.0, 0.5])
    >>> Y = np.vstack((M @ x_true, np.linalg.matrix_power(M, 2) @ x_true))
    >>> x_b = np.array([0.5, 0.2])
    >>> B = np.eye(2)
    >>> R = 0.01 * np.eye(2)
    >>> x0 = four_dvar_linear(x_b, B, Y, H, R, M)
    >>> float(np.linalg.norm(x0 - x_true)) < 0.3
    True
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from naviertwin._native import _kernels
from naviertwin.utils.logger import get_logger

logger = get_logger(__name__)


def four_dvar_linear(
    x_b: NDArray[np.float64],
    B: NDArray[np.float64],
    Y: NDArray[np.float64],
    H: NDArray[np.float64],
    R: NDArray[np.float64],
    M: NDArray[np.float64],
) -> NDArray[np.float64]:
    """선형 4D-Var 해석해.

    비용함수의 1차 조건:
        [B^{-1} + Σ_k (M^k)ᵀ Hᵀ R^{-1} H M^k] · δ = Σ_k (M^k)ᵀ Hᵀ R^{-1} (y_k - H M^k x_b)

    x_0 = x_b + δ.

    Args:
        x_b: (n,) 배경 상태.
        B: (n, n) 배경 오차 공분산.
        Y: (K, m) 관측 (t=1..K).
        H: (m, n) 관측 연산자.
        R: (m, m) 관측 오차 공분산.
        M: (n, n) 시간 발전자.

    Returns:
        최적 초기 상태 x_0 (n,).
    """
    if _kernels is None:
        raise ImportError("naviertwin._native._kernels is required by four_dvar_linear")
    x_b = np.asarray(x_b, dtype=np.float64).ravel()
    B = np.asarray(B, dtype=np.float64)
    Y = np.asarray(Y, dtype=np.float64)
    H = np.asarray(H, dtype=np.float64)
    R = np.asarray(R, dtype=np.float64)
    M = np.asarray(M, dtype=np.float64)

    n = x_b.size
    if Y.ndim != 2:
        raise ValueError(f"Y (K, m) 2D 필요: {Y.shape}")
    K = Y.shape[0]

    B_inv = np.linalg.inv(B)
    R_inv = np.linalg.inv(R)

    # A = B^{-1} + Σ (M^k)ᵀ Hᵀ R^{-1} H M^k
    A = B_inv.copy()
    b = np.zeros(n)
    Mk = np.eye(n)
    k = 0
    while k < K:
        Mk = M @ Mk  # M^{k+1}
        HM = H @ Mk
        innov = Y[k] - HM @ x_b
        A += HM.T @ R_inv @ HM
        b += HM.T @ R_inv @ innov
        k += 1

    delta = _kernels.solve_dense(A, b)
    x0 = x_b + delta
    logger.info("4D-Var: K=%d, ||δ||=%.4g", K, float(np.linalg.norm(delta)))
    return x0


__all__ = ["four_dvar_linear"]
