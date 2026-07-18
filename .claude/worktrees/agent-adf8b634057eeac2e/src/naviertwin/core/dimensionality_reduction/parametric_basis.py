"""파라메트릭 POD — Grassmann manifold 보간 + 기저 정렬.

여러 파라미터 점에서 학습된 POD 기저를 새 파라미터에 대해 보간한다.
Amsallem & Farhat (2008)의 Grassmann tangent-space 보간 사용.

References:
    Amsallem, D. & Farhat, C., "Interpolation Method: Adapting
    Reduced-Order Models and Application to Aeroelasticity",
    AIAA Journal 46(7):1803-1813, 2008.

Examples:
    >>> import numpy as np
    >>> rng = np.random.default_rng(0)
    >>> # 3개 파라미터 점에서 학습된 (n=20, r=4) 기저
    >>> bases = []
    >>> idx = 0
    >>> while idx < 3:
    ...     bases.append(np.linalg.qr(rng.standard_normal((20, 4)))[0])
    ...     idx += 1
    >>> from naviertwin.core.dimensionality_reduction.parametric_basis import (
    ...     align_bases
    ... )
    >>> aligned = align_bases(bases)
    >>> aligned[0].shape
    (20, 4)
"""

from __future__ import annotations

import numpy as np
from numpy.linalg import svd as _svd
from numpy.typing import NDArray

from naviertwin.utils.logger import get_logger

logger = get_logger(__name__)


def align_bases(
    bases: list[NDArray[np.float64]],
    reference: NDArray[np.float64] | None = None,
) -> list[NDArray[np.float64]]:
    """Procrustes 정렬 — 모든 기저를 reference에 맞춰 회전.

    각 기저를 reference의 부호와 일치하도록 SVD 기반 회전.

    Args:
        bases: list of (n, r) — 같은 차원의 정규 기저들.
        reference: (n, r) 기준 기저. None이면 첫 기저.

    Returns:
        같은 형상의 정렬된 기저 리스트.

    Raises:
        ValueError: 기저 형상 불일치.
    """
    if not bases:
        return []
    shape0 = bases[0].shape
    idx = 0
    while idx < len(bases):
        b = bases[idx]
        if b.shape != shape0:
            raise ValueError(
                f"all bases must have same shape, got {b.shape} vs {shape0}"
            )
        idx += 1
    ref = bases[0] if reference is None else np.asarray(reference, dtype=np.float64)
    if ref.shape != shape0:
        raise ValueError(
            f"reference shape {ref.shape} != {shape0}"
        )

    aligned = []
    idx = 0
    while idx < len(bases):
        b = bases[idx]
        M = ref.T @ b
        U, _, Vt = _svd(M, full_matrices=False)
        R = Vt.T @ U.T
        aligned.append(b @ R)
        idx += 1
    return aligned


def grassmann_log(
    Y0: NDArray[np.float64],
    Y1: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Grassmann manifold log 매핑 — Y0의 tangent 공간에서 Y1을 표현.

    Γ = log_{Y0}(Y1) = U arctan(Σ) Vᵀ, where (I - Y0 Y0ᵀ) Y1 (Y0ᵀ Y1)⁻¹ = U Σ Vᵀ.

    Args:
        Y0: (n, r) 기준 기저 (정규 직교).
        Y1: (n, r) 다른 기저.

    Returns:
        (n, r) tangent vector.

    Raises:
        ValueError: 형상 불일치.
    """
    Y0 = np.asarray(Y0, dtype=np.float64)
    Y1 = np.asarray(Y1, dtype=np.float64)
    if Y0.shape != Y1.shape:
        raise ValueError(f"shape mismatch: {Y0.shape} vs {Y1.shape}")

    YtY = Y0.T @ Y1
    # (I - Y0 Y0^T) Y1 (Y0^T Y1)^-1
    try:
        inv = np.linalg.inv(YtY)
    except np.linalg.LinAlgError:
        inv = np.linalg.pinv(YtY)
    M = (Y1 - Y0 @ YtY) @ inv
    U, s, Vt = _svd(M, full_matrices=False)
    s = np.clip(s, -1.0 + 1e-12, 1.0 - 1e-12)
    angles = np.arctan(s)
    return U @ np.diag(angles) @ Vt


def grassmann_exp(
    Y0: NDArray[np.float64],
    Gamma: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Grassmann exp — tangent vector Γ를 manifold로 retraction.

    Y(t) = (Y0 V cos(Σ) + U sin(Σ)) Vᵀ where Γ = U Σ Vᵀ.

    Args:
        Y0: (n, r) 기준 기저.
        Gamma: (n, r) tangent vector.

    Returns:
        (n, r) 새 manifold 점.

    Raises:
        ValueError: 형상 불일치.
    """
    Y0 = np.asarray(Y0, dtype=np.float64)
    Gamma = np.asarray(Gamma, dtype=np.float64)
    if Y0.shape != Gamma.shape:
        raise ValueError(f"shape mismatch: {Y0.shape} vs {Gamma.shape}")
    U, s, Vt = _svd(Gamma, full_matrices=False)
    Y = Y0 @ Vt.T @ np.diag(np.cos(s)) @ Vt + U @ np.diag(np.sin(s)) @ Vt
    # 정규화 (수치적 직교성 회복)
    Q, _ = np.linalg.qr(Y)
    return Q


def linear_interpolate_bases(
    bases: list[NDArray[np.float64]],
    params: NDArray[np.float64],
    target: float,
) -> NDArray[np.float64]:
    """1D 파라미터 grid에서 두 인접 기저를 Grassmann 보간.

    Args:
        bases: list of (n, r) 기저.
        params: (M,) 단조 증가 파라미터 좌표.
        target: 평가할 새 파라미터.

    Returns:
        (n, r) 보간된 기저.

    Raises:
        ValueError: 입력 오류.
    """
    if len(bases) != len(params):
        raise ValueError(
            f"bases ({len(bases)}) and params ({len(params)}) length mismatch"
        )
    if len(bases) < 2:
        raise ValueError("need at least 2 bases to interpolate")
    p = np.asarray(params, dtype=np.float64)
    if not np.all(np.diff(p) > 0):
        raise ValueError("params must be strictly increasing")

    # 정렬
    aligned = align_bases(bases)

    # 클램프 + 인접 인덱스
    if target <= p[0]:
        return aligned[0].copy()
    if target >= p[-1]:
        return aligned[-1].copy()

    idx = int(np.searchsorted(p, target))
    p0, p1 = p[idx - 1], p[idx]
    t = (target - p0) / (p1 - p0)
    Y0 = aligned[idx - 1]
    Y1 = aligned[idx]

    Gamma = grassmann_log(Y0, Y1)
    return grassmann_exp(Y0, t * Gamma)


def basis_distance_curve(
    bases: list[NDArray[np.float64]],
) -> NDArray[np.float64]:
    """인접 기저들 사이의 chordal Grassmann 거리 시퀀스.

    Args:
        bases: list of (n, r).

    Returns:
        (M-1,) 거리 배열.

    Raises:
        ValueError: 형상 불일치.
    """
    if len(bases) < 2:
        return np.array([])
    shape0 = bases[0].shape
    out = np.zeros(len(bases) - 1)
    i = 0
    while i < len(bases) - 1:
        if bases[i + 1].shape != shape0:
            raise ValueError(
                f"basis {i + 1} shape {bases[i + 1].shape} != {shape0}"
            )
        M = bases[i].T @ bases[i + 1]
        s = _svd(M, compute_uv=False)
        s = np.clip(s, -1.0, 1.0)
        angles = np.arccos(s)
        out[i] = float(np.sqrt(np.sum(np.sin(angles) ** 2)))
        i += 1
    return out


__all__ = [
    "align_bases",
    "grassmann_log",
    "grassmann_exp",
    "linear_interpolate_bases",
    "basis_distance_curve",
]
