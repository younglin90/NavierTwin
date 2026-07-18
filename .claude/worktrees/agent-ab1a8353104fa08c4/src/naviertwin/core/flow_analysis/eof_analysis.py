"""경험적 직교함수 (Empirical Orthogonal Functions, EOF) 분석.

다변수 시공간 데이터에서 지배적 변동 모드 추출. POD와 등가이지만
변수 표준화/면적 가중/북-반구식 EOF1 부호 정규화 등 기상학/해양학 표준
관행을 포함.

상용 툴 대응:
    - NCL/CDO: eofunc_n
    - MATLAB Climate Data Toolbox: eof
    - PyClimat: empirical_orthogonal_functions

References:
    Lorenz, E.N., "Empirical orthogonal functions and statistical weather
    prediction", MIT, 1956.
    von Storch & Zwiers, "Statistical Analysis in Climate Research", 1999.

Examples:
    >>> import numpy as np
    >>> rng = np.random.default_rng(0)
    >>> X = rng.standard_normal((100, 30))  # (n_t, n_space)
    >>> from naviertwin.core.flow_analysis.eof_analysis import eof_decomposition
    >>> eofs, pcs, var_explained = eof_decomposition(X, n_modes=5)
    >>> eofs.shape
    (30, 5)
"""

from __future__ import annotations

import numpy as np
from numpy.linalg import svd as _svd
from numpy.typing import NDArray

from naviertwin.utils.logger import get_logger

logger = get_logger(__name__)


def eof_decomposition(
    X: NDArray[np.float64],
    n_modes: int = 10,
    weights: NDArray[np.float64] | None = None,
    standardize: bool = False,
    sign_convention: bool = True,
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """EOF 분해.

    Args:
        X: (n_t, n_space) 시간-공간 배열.
        n_modes: 추출할 모드 수.
        weights: (n_space,) 면적 또는 셀-볼륨 가중. None이면 균일.
        standardize: True면 각 공간 위치에서 분산 정규화 (correlation EOF).
        sign_convention: True면 EOF1 최대 절댓값 위치에서 양으로 정규화.

    Returns:
        (eofs, pcs, var_explained):
            eofs: (n_space, n_modes) 공간 패턴.
            pcs: (n_t, n_modes) 시간 계수 (Principal Components).
            var_explained: (n_modes,) 각 모드의 분산 비율.

    Raises:
        ValueError: X가 2D 아님 또는 형상 불일치.
    """
    X = np.asarray(X, dtype=np.float64)
    if X.ndim != 2:
        raise ValueError(f"X must be 2D (n_t, n_space), got {X.shape}")
    n_t, n_x = X.shape
    if n_modes <= 0:
        raise ValueError(f"n_modes must be > 0, got {n_modes}")
    n_modes = min(n_modes, n_t, n_x)

    # 시간 평균 제거 (anomaly)
    X_mean = X.mean(axis=0, keepdims=True)
    A = X - X_mean

    if standardize:
        sigma = A.std(axis=0, keepdims=True) + 1e-30
        A = A / sigma

    # 면적 가중
    if weights is not None:
        w = np.asarray(weights, dtype=np.float64)
        if w.shape != (n_x,):
            raise ValueError(
                f"weights shape {w.shape} != ({n_x},)"
            )
        if np.any(w < 0):
            raise ValueError("weights must be non-negative")
        sw = np.sqrt(w)[None, :]
        A_w = A * sw
    else:
        sw = None
        A_w = A

    # SVD: A_w = U S V^T → EOFs는 V의 열 (가중치 제거 후)
    U, s, Vt = _svd(A_w, full_matrices=False)
    eofs_w = Vt[:n_modes].T  # (n_space, n_modes)

    if sw is not None:
        eofs = eofs_w / np.maximum(sw.ravel()[:, None], 1e-30)
    else:
        eofs = eofs_w

    # PC 시간 계수: A @ eofs (가중치 없이 투영)
    pcs = A @ eofs[:, :n_modes]

    # 분산 비율
    s_total = float(np.sum(s ** 2))
    if s_total < 1e-30:
        var_explained = np.zeros(n_modes)
    else:
        var_explained = (s[:n_modes] ** 2) / s_total

    # 부호 컨벤션: EOF1 최대 절댓값 위치가 양수
    if sign_convention:
        mode_idx = np.arange(n_modes)
        idx_max = np.argmax(np.abs(eofs[:, :n_modes]), axis=0)
        signs = np.where(eofs[idx_max, mode_idx] < 0.0, -1.0, 1.0)
        eofs[:, :n_modes] *= signs
        pcs[:, :n_modes] *= signs

    logger.info("EOF 분해 완료: %d 모드, 누적 분산=%.1f%%",
                n_modes, var_explained.sum() * 100)
    return eofs, pcs, var_explained


def reconstruct_from_eof(
    eofs: NDArray[np.float64],
    pcs: NDArray[np.float64],
    mean: NDArray[np.float64] | None = None,
) -> NDArray[np.float64]:
    """EOF 절단 재구성: X̂ = mean + PC @ EOFᵀ.

    Args:
        eofs: (n_space, n_modes).
        pcs: (n_t, n_modes).
        mean: (n_space,) 평균. None이면 0.

    Returns:
        (n_t, n_space) 재구성 필드.

    Raises:
        ValueError: 형상 불일치.
    """
    eofs = np.asarray(eofs, dtype=np.float64)
    pcs = np.asarray(pcs, dtype=np.float64)
    if eofs.ndim != 2 or pcs.ndim != 2:
        raise ValueError(
            f"eofs/pcs must be 2D, got {eofs.shape}, {pcs.shape}"
        )
    if eofs.shape[1] != pcs.shape[1]:
        raise ValueError(
            f"n_modes mismatch: eofs={eofs.shape[1]} vs pcs={pcs.shape[1]}"
        )

    rec = pcs @ eofs.T
    if mean is not None:
        rec = rec + np.asarray(mean, dtype=np.float64)[None, :]
    return rec


def north_significance_test(
    var_explained: NDArray[np.float64],
    n_t: int,
) -> NDArray[np.float64]:
    """North et al. 1982 EOF 분리 가능성 검정.

    각 모드 i 와 인접 모드의 분산 차가 √(2/n_t) · λ_i 이상이면 잘 분리됨.

    Args:
        var_explained: (n_modes,) 분산 비율.
        n_t: 시간 표본 수.

    Returns:
        (n_modes,) 각 모드의 표준 오차 추정. 인접 모드와의 차가
        이보다 커야 통계적으로 분리 가능.

    Raises:
        ValueError: n_t < 2.
    """
    var_explained = np.asarray(var_explained, dtype=np.float64)
    if n_t < 2:
        raise ValueError(f"n_t must be >= 2, got {n_t}")
    return var_explained * np.sqrt(2.0 / n_t)


def varimax_rotation(
    eofs: NDArray[np.float64],
    n_iter: int = 100,
    tol: float = 1e-6,
) -> NDArray[np.float64]:
    """Varimax 회전 — EOF의 공간 국소화를 강화 (희소성 증가).

    Kaiser 1958: 분산 합 최대화. 각 EOF가 더 명확한 spatial pattern을 갖도록.

    Args:
        eofs: (n_space, n_modes) EOF.
        n_iter: 최대 반복.
        tol: 수렴 허용 오차.

    Returns:
        회전된 EOF (같은 형상).

    Raises:
        ValueError: 형상 오류.
    """
    A = np.asarray(eofs, dtype=np.float64)
    if A.ndim != 2:
        raise ValueError(f"eofs must be 2D, got {A.shape}")
    n, k = A.shape
    if k < 2:
        return A.copy()

    R = np.eye(k)
    d = 0.0
    iteration = 0
    while iteration < n_iter:
        d_old = d
        Lambda = A @ R
        # SVD 기반 업데이트 (Kaiser)
        u, s, vh = _svd(
            A.T @ (Lambda ** 3 - (Lambda @ np.diag(np.diag(Lambda.T @ Lambda))) / n),
            full_matrices=False,
        )
        R = u @ vh
        d = float(np.sum(s))
        iteration += 1
        if abs(d - d_old) < tol:
            break

    return A @ R


__all__ = [
    "eof_decomposition",
    "reconstruct_from_eof",
    "north_significance_test",
    "varimax_rotation",
]
