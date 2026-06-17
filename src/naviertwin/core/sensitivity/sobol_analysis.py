"""Sobol 전역 민감도 분석 (Saltelli 샘플링).

SALib 이 설치된 경우 그 백엔드를 사용, 아니면 직접 구현한 Saltelli + Sobol
인덱스 공식으로 계산.

References:
    Sobol' 2001; Saltelli et al. 2010; Herman & Usher, JOSS 2017 (SALib).

Examples:
    >>> import numpy as np
    >>> from naviertwin.core.sensitivity.sobol_analysis import (
    ...     saltelli_sample, sobol_indices,
    ... )
    >>> bounds = np.array([[0, 1], [0, 1], [0, 1]], dtype=float)
    >>> X = saltelli_sample(bounds, n_base=256, seed=0)
    >>> def fmodel(v):
    ...     return np.sin(v[:, 0]) + v[:, 1] * v[:, 2]
    >>> Y = fmodel(X)
    >>> S = sobol_indices(Y, n_params=3)
    >>> sorted(S.keys())
    ['S1', 'ST']
"""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray

from naviertwin.utils.logger import get_logger

logger = get_logger(__name__)


def saltelli_sample(
    bounds: NDArray[np.float64],
    n_base: int = 512,
    seed: int | None = None,
) -> NDArray[np.float64]:
    """Saltelli 행렬을 생성한다.

    Args:
        bounds: (n_params, 2) 각 파라미터 [low, high].
        n_base: 기본 샘플 수. 총 샘플 수 = n_base × (2·n_params + 2).
        seed: 재현용 seed.

    Returns:
        (n_total, n_params) 샘플 배열.
    """
    bounds = np.asarray(bounds, dtype=np.float64)
    if bounds.ndim != 2 or bounds.shape[1] != 2:
        raise ValueError(f"bounds shape={bounds.shape} != (n, 2)")
    rng = np.random.default_rng(seed)
    n = bounds.shape[0]

    # 2 개의 독립 샘플 행렬 A, B (n_base, n)
    A = rng.random((n_base, n))
    B = rng.random((n_base, n))

    # A_B^{(i)}: A 에서 i 번째 컬럼만 B 로 교체
    idx = np.arange(n)
    AB = np.broadcast_to(A, (n, n_base, n)).copy()
    AB[idx, :, idx] = B.T
    # BA^{(i)}: B 에서 i 번째 컬럼만 A 로 교체 (Total effect)
    BA = np.broadcast_to(B, (n, n_base, n)).copy()
    BA[idx, :, idx] = A.T

    X01 = np.concatenate((A[None, :, :], B[None, :, :], AB, BA), axis=0).reshape(-1, n)
    # bounds 로 스케일
    lows = bounds[:, 0]
    highs = bounds[:, 1]
    X = lows + X01 * (highs - lows)
    return X.astype(np.float64)


def sobol_indices(
    Y: NDArray[np.float64], n_params: int
) -> dict[str, NDArray[np.float64]]:
    """Saltelli 순서로 평가된 Y 에서 S1, ST 를 계산한다.

    Args:
        Y: 모델 응답. 길이 = n_base × (2·n_params + 2).
        n_params: 파라미터 수.

    Returns:
        {"S1": (n_params,), "ST": (n_params,)}
    """
    Y = np.asarray(Y, dtype=np.float64).ravel()
    total = Y.size
    n_base = total // (2 * n_params + 2)
    if n_base * (2 * n_params + 2) != total:
        raise ValueError(
            f"Y 길이({total}) != n_base * (2*n+2) — saltelli_sample 에서 생성한 Y 여야 함"
        )

    Y_A = Y[:n_base]
    Y_B = Y[n_base : 2 * n_base]
    Y_AB = Y[2 * n_base : (n_params + 2) * n_base].reshape(n_params, n_base)
    Y_BA = Y[(n_params + 2) * n_base :].reshape(n_params, n_base)

    var_Y = float(np.var(np.concatenate([Y_A, Y_B])))
    if var_Y == 0:
        var_Y = 1e-30

    # Saltelli 2010 estimator
    S1 = np.mean(Y_B[None, :] * (Y_AB - Y_A[None, :]), axis=1) / var_Y
    ST = 0.5 * np.mean((Y_A[None, :] - Y_BA) ** 2, axis=1) / var_Y

    return {"S1": S1, "ST": ST}


def sobol_with_salib(
    problem: dict[str, Any], n_base: int = 512, model: Any = None
) -> Any:
    """SALib 이 있으면 그걸로 Sobol 분석을 수행한다 (편의 래퍼).

    Args:
        problem: SALib 형식 dict (num_vars, names, bounds).
        n_base: 기본 샘플 수.
        model: callable(X) → Y 모델.

    Returns:
        SALib analyze.sobol 결과 dict.

    Raises:
        RuntimeError: SALib 미설치 시.
    """
    try:
        from SALib.analyze import sobol as sobol_salib
        from SALib.sample import saltelli as sampling
    except ImportError as exc:
        raise RuntimeError(
            "SALib 설치 필요: pip install SALib"
        ) from exc

    X = sampling.sample(problem, n_base)
    if model is None:
        raise ValueError("model callable 이 필요합니다")
    Y = np.asarray(model(X), dtype=np.float64).ravel()
    return sobol_salib.analyze(problem, Y)


__all__ = ["saltelli_sample", "sobol_indices", "sobol_with_salib"]
