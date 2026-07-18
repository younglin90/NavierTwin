"""Gappy POD (Everson-Sirovich 1995) — 결측 데이터 복원.

CFD 측정/실험에서 일부 센서가 누락되었거나, 부분 시야만 관측 가능할 때,
미리 학습된 POD 기저로 누락 영역을 재구성. ROM-기반 데이터 보완의 표준.

References:
    Everson, R. & Sirovich, L., "Karhunen-Loève procedure: gappy data",
    JOSA A 12(8):1657-1664, 1995.

Examples:
    >>> import numpy as np
    >>> from numpy.linalg import svd as _svd
    >>> rng = np.random.default_rng(0)
    >>> X_full = rng.standard_normal((50, 30))
    >>> U, _, _ = _svd(X_full, full_matrices=False)
    >>> mask = np.ones(50, dtype=bool); mask[10:20] = False
    >>> from naviertwin.core.dimensionality_reduction.gappy_pod import (
    ...     gappy_reconstruct
    ... )
    >>> x_partial = X_full[:, 0]
    >>> x_full_recovered = gappy_reconstruct(U[:, :5], x_partial, mask)
    >>> x_full_recovered.shape
    (50,)
"""

from __future__ import annotations

import numpy as np
from numpy.linalg import svd as _svd
from numpy.typing import NDArray

from naviertwin.utils.logger import get_logger

logger = get_logger(__name__)


def gappy_reconstruct(
    basis: NDArray[np.float64],
    partial: NDArray[np.float64],
    mask: NDArray[np.bool_],
) -> NDArray[np.float64]:
    """결측 데이터의 POD 기반 복원.

    Args:
        basis: (n_full, r) 사전 학습된 기저.
        partial: (n_full,) 또는 (n_full, n_samples) 데이터; mask=False 영역은 무시.
        mask: (n_full,) True = 관측, False = 결측.

    Returns:
        같은 형상의 완전 복원 배열.

    Raises:
        ValueError: 형상 불일치.
    """
    V = np.asarray(basis, dtype=np.float64)
    p = np.asarray(partial, dtype=np.float64)
    m = np.asarray(mask, dtype=bool)

    if V.ndim != 2:
        raise ValueError(f"basis must be 2D, got {V.shape}")
    n_full = V.shape[0]
    if m.shape != (n_full,):
        raise ValueError(
            f"mask shape {m.shape} != ({n_full},)"
        )

    if p.ndim == 1:
        if p.shape[0] != n_full:
            raise ValueError(
                f"partial length {p.shape[0]} != n_full {n_full}"
            )
        return _solve_single(V, p, m)
    if p.ndim == 2:
        if p.shape[0] != n_full:
            raise ValueError(
                f"partial rows {p.shape[0]} != n_full {n_full}"
            )
        if not m.any():
            return np.zeros_like(p)
        V_obs = V[m, :]
        alpha, _, _, _ = np.linalg.lstsq(V_obs, p[m, :], rcond=None)
        return V @ alpha
    raise ValueError(f"partial must be 1D or 2D, got {p.shape}")


def _solve_single(
    V: NDArray[np.float64],
    p: NDArray[np.float64],
    m: NDArray[np.bool_],
) -> NDArray[np.float64]:
    """단일 벡터 gappy 복원.

    α = (V_obs^T V_obs)^-1 V_obs^T p_obs.
    Reconstruction = V α (전체 위치).
    """
    if not m.any():
        return np.zeros_like(p)
    V_obs = V[m, :]
    p_obs = p[m]
    # least squares
    alpha, _, _, _ = np.linalg.lstsq(V_obs, p_obs, rcond=None)
    return V @ alpha


def gappy_coefficients(
    basis: NDArray[np.float64],
    partial: NDArray[np.float64],
    mask: NDArray[np.bool_],
) -> NDArray[np.float64]:
    """관측치만으로 POD 계수 α를 추정한다.

    Args:
        basis: (n_full, r) 기저.
        partial: (n_full,) 데이터.
        mask: (n_full,) 부울 마스크.

    Returns:
        (r,) 계수.

    Raises:
        ValueError: 형상 오류.
    """
    V = np.asarray(basis, dtype=np.float64)
    p = np.asarray(partial, dtype=np.float64).ravel()
    m = np.asarray(mask, dtype=bool).ravel()
    if V.ndim != 2:
        raise ValueError(f"basis must be 2D, got {V.shape}")
    n_full = V.shape[0]
    if p.shape[0] != n_full or m.shape[0] != n_full:
        raise ValueError(
            f"partial/mask shape mismatch: {p.shape}, {m.shape} vs {n_full}"
        )
    if not m.any():
        return np.zeros(V.shape[1])
    V_obs = V[m, :]
    p_obs = p[m]
    alpha, _, _, _ = np.linalg.lstsq(V_obs, p_obs, rcond=None)
    return alpha


def gappy_iter(
    X: NDArray[np.float64],
    mask: NDArray[np.bool_],
    n_modes: int,
    max_iter: int = 50,
    tol: float = 1e-6,
) -> NDArray[np.float64]:
    """반복 Gappy POD — 결측 데이터를 0으로 초기화 후, POD ↔ 보완 반복.

    Args:
        X: (n_full, n_samples) 데이터, 결측 위치는 임의 값.
        mask: (n_full, n_samples) 또는 (n_full,) 부울 마스크.
        n_modes: POD 모드 수.
        max_iter: 최대 반복.
        tol: 수렴 임계 (L2 변화율).

    Returns:
        결측 영역이 채워진 (n_full, n_samples) 배열.

    Raises:
        ValueError: X가 2D 아님 또는 mask 형상 오류.
    """
    X = np.asarray(X, dtype=np.float64)
    if X.ndim != 2:
        raise ValueError(f"X must be 2D, got {X.shape}")
    n_full, n_samples = X.shape
    m = np.asarray(mask, dtype=bool)
    if m.shape == (n_full,):
        m = np.broadcast_to(m[:, None], X.shape).copy()
    elif m.shape != X.shape:
        raise ValueError(
            f"mask shape {m.shape} incompatible with X {X.shape}"
        )
    if n_modes <= 0:
        raise ValueError(f"n_modes > 0, got {n_modes}")

    # 초기화: 결측을 평균으로
    X_filled = X.copy()
    obs_counts = m.sum(axis=0)
    obs_sums = np.where(m, X_filled, 0.0).sum(axis=0)
    fill_values = np.divide(
        obs_sums,
        obs_counts,
        out=np.zeros(n_samples, dtype=np.float64),
        where=obs_counts > 0,
    )
    X_filled = np.where(m, X_filled, fill_values[None, :])

    prev_err = np.inf
    it = 0
    while it < max_iter:
        X_centered = X_filled - X_filled.mean(axis=1, keepdims=True)
        U, _, _ = _svd(X_centered, full_matrices=False)
        r = min(n_modes, U.shape[1])
        V = U[:, :r]
        # 각 컬럼을 gappy reconstruct
        new_X = X_filled.copy()
        j = 0
        while j < n_samples:
            new_X[:, j] = _solve_single(V, X[:, j], m[:, j])
            j += 1
        # 수렴 체크
        diff = float(np.linalg.norm(new_X - X_filled))
        # 결측 부분만 새 값 적용
        X_filled = np.where(m, X, new_X)
        if abs(prev_err - diff) < tol:
            logger.info("Gappy POD converged at iter %d (diff=%.4g)", it, diff)
            break
        prev_err = diff
        it += 1

    return X_filled


def reconstruction_error(
    X_true: NDArray[np.float64],
    X_filled: NDArray[np.float64],
    mask: NDArray[np.bool_],
) -> float:
    """결측 영역의 RMSE만 평가 (관측 영역 제외).

    Args:
        X_true: 진짜 값.
        X_filled: 복원된 배열.
        mask: True = 관측.

    Returns:
        결측 영역 RMSE.

    Raises:
        ValueError: 형상 불일치.
    """
    X_true = np.asarray(X_true, dtype=np.float64)
    X_filled = np.asarray(X_filled, dtype=np.float64)
    m = np.asarray(mask, dtype=bool)
    if X_true.shape != X_filled.shape or m.shape != X_true.shape:
        raise ValueError(
            f"shape mismatch: {X_true.shape}, {X_filled.shape}, {m.shape}"
        )
    missing = ~m
    if not missing.any():
        return 0.0
    diff = X_true[missing] - X_filled[missing]
    return float(np.sqrt(np.mean(diff ** 2)))


__all__ = [
    "gappy_reconstruct",
    "gappy_coefficients",
    "gappy_iter",
    "reconstruction_error",
]
