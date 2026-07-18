"""POD/SVD 모드 추적 — 부분공간 각도 + drift 검출.

서로 다른 시점/조건에서 학습된 두 ROM 기저의 닮음을 정량화. 시간 경과에
따른 모드 drift, 다른 파라미터 영역으로의 일반화 가능성 평가.

상용 툴 대응:
    - pyMOR: subspace_angle / GramSchmidt
    - MATLAB: subspace, orth
    - 학술: Bjorck & Golub, "Numerical Methods: Computing Angles
      Between Linear Subspaces", Math. Comp., 1973.

Examples:
    >>> import numpy as np
    >>> # 두 동일한 부분공간 → 각도 0
    >>> A = np.eye(5)[:, :3]
    >>> B = np.eye(5)[:, :3]
    >>> from naviertwin.core.dimensionality_reduction.mode_tracking import (
    ...     subspace_angles
    ... )
    >>> angles = subspace_angles(A, B)
    >>> max(abs(angles)) < 1e-10
    True
"""

from __future__ import annotations

import numpy as np
from numpy.linalg import svd as _svd
from numpy.typing import NDArray

from naviertwin.utils.logger import get_logger

logger = get_logger(__name__)


def subspace_angles(
    A: NDArray[np.float64],
    B: NDArray[np.float64],
) -> NDArray[np.float64]:
    """두 부분공간 사이의 정준 각도 (canonical angles, principal angles).

    A, B의 직교화 → SVD(Aᵀ B) → singular value들이 cos(angles).

    Args:
        A: (n, k_A) 첫 번째 부분공간 기저.
        B: (n, k_B) 두 번째 부분공간 기저.

    Returns:
        (min(k_A, k_B),) 오름차순 정렬된 각도 [라디안].

    Raises:
        ValueError: 첫 번째 차원 불일치.
    """
    A = np.asarray(A, dtype=np.float64)
    B = np.asarray(B, dtype=np.float64)
    if A.ndim != 2 or B.ndim != 2:
        raise ValueError(
            f"A and B must be 2D, got {A.shape}, {B.shape}"
        )
    if A.shape[0] != B.shape[0]:
        raise ValueError(
            f"row count mismatch: {A.shape[0]} vs {B.shape[0]}"
        )

    # 직교화
    Qa, _ = np.linalg.qr(A)
    Qb, _ = np.linalg.qr(B)

    M = Qa.T @ Qb
    s = _svd(M, compute_uv=False)
    s = np.clip(s, -1.0, 1.0)
    angles = np.arccos(s)
    return np.sort(angles)


def grassmann_distance(
    A: NDArray[np.float64],
    B: NDArray[np.float64],
) -> float:
    """Grassmann 거리 = ‖angles‖₂ (대 측정 거리).

    Args:
        A, B: 두 부분공간 기저.

    Returns:
        Grassmann 거리.
    """
    angles = subspace_angles(A, B)
    return float(np.linalg.norm(angles))


def subspace_distance_chordal(
    A: NDArray[np.float64],
    B: NDArray[np.float64],
) -> float:
    """Chordal 거리 = ‖sin(angles)‖₂ ∈ [0, √(min(k_A, k_B))].

    Args:
        A, B: 부분공간 기저.

    Returns:
        Chordal 거리.
    """
    angles = subspace_angles(A, B)
    return float(np.linalg.norm(np.sin(angles)))


def mode_alignment_matrix(
    A: NDArray[np.float64],
    B: NDArray[np.float64],
) -> NDArray[np.float64]:
    """A와 B 모드 간 cosine 유사도 행렬 |a_i · b_j|.

    Args:
        A: (n, k_A).
        B: (n, k_B).

    Returns:
        (k_A, k_B) 절댓값 cosine 유사도.
    """
    A = np.asarray(A, dtype=np.float64)
    B = np.asarray(B, dtype=np.float64)
    if A.shape[0] != B.shape[0]:
        raise ValueError("row count mismatch")
    norm_a = np.linalg.norm(A, axis=0) + 1e-30
    norm_b = np.linalg.norm(B, axis=0) + 1e-30
    return np.abs(A.T @ B) / (norm_a[:, None] * norm_b[None, :])


def best_match_assignment(
    similarity: NDArray[np.float64],
) -> NDArray[np.intp]:
    """Greedy maximum-weight 매칭 — A 모드별 최적 B 모드 인덱스.

    Args:
        similarity: (k_A, k_B) 유사도 행렬.

    Returns:
        (k_A,) 정수 인덱스 (B 모드 인덱스). -1 = 매칭 안 됨.
    """
    sim = np.asarray(similarity, dtype=np.float64).copy()
    if sim.ndim != 2:
        raise ValueError(f"similarity must be 2D, got {sim.shape}")
    k_A, k_B = sim.shape
    out = np.full(k_A, -1, dtype=np.intp)
    used_b = set()
    # greedy: 가장 큰 유사도 순으로
    flat_idx = np.argsort(sim.ravel())[::-1]
    pos = 0
    while pos < flat_idx.size:
        fi = flat_idx[pos]
        i = fi // k_B
        j = fi % k_B
        if out[i] != -1 or j in used_b:
            pos += 1
            continue
        out[i] = j
        used_b.add(j)
        if (out != -1).all():
            break
        pos += 1
    return out


def drift_score(
    basis_old: NDArray[np.float64],
    basis_new: NDArray[np.float64],
) -> float:
    """시간 경과 모드 drift 점수 ∈ [0, 1]. 0 = 동일, 1 = 완전 직교.

    Score = (2/π) · max_angle — 최대 부분공간 각도의 정규화.

    Args:
        basis_old, basis_new: ROM 기저.

    Returns:
        drift 점수.
    """
    angles = subspace_angles(basis_old, basis_new)
    if len(angles) == 0:
        return 0.0
    return float(np.max(angles) * 2.0 / np.pi)


def proper_orthogonal_basis(
    X: NDArray[np.float64],
    n_modes: int,
    center: bool = True,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """X의 POD 기저와 특이값 반환 — mode_tracking 헬퍼.

    Args:
        X: (n_t, n_x) 또는 (n_x, n_t) (행 우선/열 우선 자동).
        n_modes: 추출할 모드 수.
        center: True면 시간 평균 제거.

    Returns:
        (basis (n_x, r), singular_values).

    Raises:
        ValueError: X가 2D 아님.
    """
    X = np.asarray(X, dtype=np.float64)
    if X.ndim != 2:
        raise ValueError(f"X must be 2D, got {X.shape}")
    # 더 큰 차원이 공간 (n_x > n_t 가정)
    if X.shape[0] < X.shape[1]:
        X = X.T  # (n_x, n_t)
    if center:
        X = X - X.mean(axis=1, keepdims=True)
    U, s, _ = _svd(X, full_matrices=False)
    r = min(n_modes, U.shape[1])
    return U[:, :r], s[:r]


__all__ = [
    "subspace_angles",
    "grassmann_distance",
    "subspace_distance_chordal",
    "mode_alignment_matrix",
    "best_match_assignment",
    "drift_score",
    "proper_orthogonal_basis",
]
