"""ROM 인증 메트릭 — a posteriori 오차 추정 + 신뢰 구간 + 검증 통계.

ROM(POD/PCA) 예측의 신뢰성을 데이터 기반으로 정량화. 새 입력에 대한
모델 출력이 얼마나 확실한지, 학습 데이터에서 얼마나 멀리 외삽하는지 진단.

상용 툴 대응:
    - Ansys Discovery: ROM Confidence Score
    - pyMOR: residual-based a posteriori error bounds
    - 학술: Quarteroni et al., "Reduced Basis Methods: Partial Differential
      Equations", Springer 2016, §3.5.

Examples:
    >>> import numpy as np
    >>> from numpy.linalg import svd as _svd
    >>> rng = np.random.default_rng(0)
    >>> X = rng.standard_normal((50, 30))
    >>> U, s, _ = _svd(X, full_matrices=False)
    >>> from naviertwin.core.dimensionality_reduction.rom_certification import (
    ...     reconstruction_residual
    ... )
    >>> r = reconstruction_residual(X, U[:, :5])
    >>> r.shape
    (50,)
"""

from __future__ import annotations

import numpy as np
from numpy.linalg import svd as _svd
from numpy.typing import NDArray

from naviertwin.utils.logger import get_logger

logger = get_logger(__name__)


def reconstruction_residual(
    X: NDArray[np.float64],
    basis: NDArray[np.float64],
) -> NDArray[np.float64]:
    """기저 절단 후 잔차 ‖x - Vᵀ V x‖₂ — 각 스냅샷별.

    Args:
        X: (n_t, n_x) 또는 (n_x,) 데이터.
        basis: (n_x, r) 직교 정규 기저.

    Returns:
        잔차 노름 — (n_t,) 또는 스칼라.

    Raises:
        ValueError: 형상 불일치.
    """
    X = np.asarray(X, dtype=np.float64)
    V = np.asarray(basis, dtype=np.float64)
    if V.ndim != 2:
        raise ValueError(f"basis must be 2D, got {V.shape}")

    if X.ndim == 1:
        if X.shape[0] != V.shape[0]:
            raise ValueError(
                f"X length {X.shape[0]} != basis n_x {V.shape[0]}"
            )
        coeffs = V.T @ X
        residual = X - V @ coeffs
        return float(np.linalg.norm(residual))

    if X.shape[1] != V.shape[0]:
        raise ValueError(
            f"X cols {X.shape[1]} != basis n_x {V.shape[0]}"
        )

    coeffs = X @ V  # (n_t, r)
    rec = coeffs @ V.T  # (n_t, n_x)
    diff = X - rec
    return np.linalg.norm(diff, axis=1)


def relative_residual(
    X: NDArray[np.float64],
    basis: NDArray[np.float64],
) -> NDArray[np.float64]:
    """상대 잔차 ‖x - V V^T x‖ / ‖x‖.

    Args:
        X: 데이터.
        basis: 기저.

    Returns:
        상대 잔차 (양수, ∈ [0, 1]).
    """
    abs_res = reconstruction_residual(X, basis)
    X = np.asarray(X, dtype=np.float64)
    if X.ndim == 1:
        norm = np.linalg.norm(X) + 1e-30
        return abs_res / norm
    norms = np.linalg.norm(X, axis=1) + 1e-30
    return abs_res / norms


def leave_one_out_score(
    X: NDArray[np.float64],
    n_modes: int,
) -> dict[str, float]:
    """LOO-CV 기반 ROM 일반화 오차 추정.

    한 스냅샷씩 빼고 나머지로 POD → 빠진 스냅샷 재구성 오차 평균.

    Args:
        X: (n_t, n_x) 스냅샷 행렬.
        n_modes: ROM 모드 수.

    Returns:
        dict {"loo_mse": float, "loo_max": float, "loo_mean_rel": float}.

    Raises:
        ValueError: X가 2D 아님 또는 n_t < 3.
    """
    X = np.asarray(X, dtype=np.float64)
    if X.ndim != 2:
        raise ValueError(f"X must be 2D, got {X.shape}")
    n_t, n_x = X.shape
    if n_t < 3:
        raise ValueError(f"need n_t >= 3 to run LOO, got {n_t}")
    if n_modes <= 0:
        raise ValueError(f"n_modes must be > 0, got {n_modes}")

    errs = []
    rel_errs = []
    i = 0
    while i < n_t:
        mask = np.ones(n_t, dtype=bool)
        mask[i] = False
        X_train = X[mask]
        X_train = X_train - X_train.mean(axis=0, keepdims=True)
        U, s, Vt = _svd(X_train.T, full_matrices=False)
        r = min(n_modes, U.shape[1])
        V = U[:, :r]
        x_test = X[i] - X[mask].mean(axis=0)
        rec = V @ (V.T @ x_test)
        err = np.linalg.norm(x_test - rec)
        errs.append(err)
        norm_x = np.linalg.norm(x_test) + 1e-30
        rel_errs.append(err / norm_x)
        i += 1

    return {
        "loo_mse": float(np.mean(np.array(errs) ** 2)),
        "loo_max": float(np.max(errs)),
        "loo_mean_rel": float(np.mean(rel_errs)),
    }


def coefficient_envelope(
    coeffs_train: NDArray[np.float64],
    new_coeff: NDArray[np.float64],
) -> dict[str, float]:
    """새 점이 학습 계수 구간 내에 있는지 검사 (외삽 진단).

    Args:
        coeffs_train: (n_t, r) 학습 시 POD 계수.
        new_coeff: (r,) 새 입력의 계수.

    Returns:
        dict with keys: max_z (모드별 최대 z-score), mahalanobis (마할라노비스 거리),
        bbox_violation_count (모드 중 학습 범위 밖 갯수).

    Raises:
        ValueError: 형상 불일치.
    """
    C = np.asarray(coeffs_train, dtype=np.float64)
    c = np.asarray(new_coeff, dtype=np.float64).ravel()
    if C.ndim != 2:
        raise ValueError(f"coeffs_train must be 2D, got {C.shape}")
    if c.shape[0] != C.shape[1]:
        raise ValueError(
            f"new_coeff length {c.shape[0]} != n_modes {C.shape[1]}"
        )

    mu = C.mean(axis=0)
    sigma = C.std(axis=0) + 1e-30
    z = (c - mu) / sigma
    max_z = float(np.max(np.abs(z)))

    # 마할라노비스
    cov = np.cov(C.T) + 1e-12 * np.eye(C.shape[1])
    diff = c - mu
    try:
        inv = np.linalg.inv(cov)
        mahal = float(np.sqrt(diff @ inv @ diff))
    except np.linalg.LinAlgError:
        mahal = float("inf")

    # bbox violation
    lo = C.min(axis=0)
    hi = C.max(axis=0)
    bbox_violations = int(np.sum((c < lo) | (c > hi)))

    return {
        "max_z": max_z,
        "mahalanobis": mahal,
        "bbox_violation_count": bbox_violations,
    }


def projection_error_bound(
    singular_values: NDArray[np.float64],
    n_modes: int,
) -> float:
    """이론적 투영 오차 상한 (Eckart-Young) ‖A - A_r‖_F = √(Σ_{i>r} σ_i²).

    Args:
        singular_values: 특이값.
        n_modes: 모드 수.

    Returns:
        절대 오차 상한.

    Raises:
        ValueError: n_modes 범위 오류.
    """
    s = np.asarray(singular_values, dtype=np.float64).ravel()
    if n_modes < 0 or n_modes > len(s):
        raise ValueError(
            f"n_modes must be in [0, {len(s)}], got {n_modes}"
        )
    return float(np.sqrt(np.sum(s[n_modes:] ** 2)))


def confidence_score(
    residual_norm: float,
    reference_norm: float,
    threshold: float = 0.05,
) -> float:
    """확신도 점수 ∈ [0, 1] — 잔차/기준이 임계값 이하면 1, 초과시 0으로 감쇠.

    Args:
        residual_norm: 잔차 노름.
        reference_norm: 기준 노름 (예: 학습 데이터 평균 노름).
        threshold: 상대 오차 임계값.

    Returns:
        신뢰도 점수.

    Raises:
        ValueError: 매개변수 오류.
    """
    if reference_norm <= 0:
        raise ValueError(f"reference_norm must be > 0, got {reference_norm}")
    if threshold <= 0:
        raise ValueError(f"threshold must be > 0, got {threshold}")
    rel = residual_norm / reference_norm
    if rel <= threshold:
        return 1.0
    # exponential decay
    return float(np.exp(-(rel - threshold) / threshold))


__all__ = [
    "reconstruction_residual",
    "relative_residual",
    "leave_one_out_score",
    "coefficient_envelope",
    "projection_error_bound",
    "confidence_score",
]
