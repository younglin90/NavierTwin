"""Reynolds 응력 텐서 비등방성 분석 — Lumley triangle, II-III invariants.

난류 비등방성을 시각화/정량화. 아이소트로픽(2-component, 1-component, ...)
에서 얼마나 떨어져 있는지 보여주는 standard turbulence diagnostic.

References:
    Lumley, J.L., "Computational Modeling of Turbulent Flows", Adv. Appl.
    Mech., 1978.
    Pope, S.B., "Turbulent Flows", §11.5, 2000.

Examples:
    >>> import numpy as np
    >>> # Isotropic stress: R = (2k/3) I → b = 0
    >>> R = (2/3) * np.eye(3)
    >>> from naviertwin.core.flow_analysis.anisotropy import anisotropy_tensor
    >>> b = anisotropy_tensor(R)
    >>> np.allclose(b, 0.0)
    True
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from naviertwin.utils.logger import get_logger

logger = get_logger(__name__)


def anisotropy_tensor(
    R: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Reynolds 응력 비등방성 텐서 b_ij = R_ij / (2k) - δ_ij/3.

    Args:
        R: (3, 3) Reynolds 응력 텐서 (대칭).

    Returns:
        (3, 3) 비등방성 텐서. trace(b) = 0.

    Raises:
        ValueError: 형상 오류.
    """
    R = np.asarray(R, dtype=np.float64)
    if R.shape != (3, 3):
        raise ValueError(f"R must be (3, 3), got {R.shape}")
    k = 0.5 * float(np.trace(R))
    if abs(k) < 1e-30:
        return np.zeros((3, 3))
    return R / (2.0 * k) - np.eye(3) / 3.0


def invariants_II_III(
    b: NDArray[np.float64],
) -> tuple[float, float]:
    """비등방성 텐서의 두 번째/세 번째 불변량.

    II = -½ trace(b²), III = ⅓ trace(b³).
    또는 II = b_ii² (대칭에 대해 = b_ij b_ij), III = b_ij b_jk b_ki / 3 (Pope 컨벤션).

    Args:
        b: (3, 3) 비등방성 텐서.

    Returns:
        (II, III).
    """
    b = np.asarray(b, dtype=np.float64)
    if b.shape != (3, 3):
        raise ValueError(f"b must be (3, 3), got {b.shape}")
    II = float(np.trace(b @ b))
    III = float(np.trace(b @ b @ b))
    return II, III


def lumley_eta_xi(
    b: NDArray[np.float64],
) -> tuple[float, float]:
    """Lumley triangle 좌표 (η, ξ).

    η² = II_b / 6, ξ³ = III_b / 6, where II_b = b_ij b_ij, III_b = b_ij b_jk b_ki.

    Args:
        b: (3, 3) 비등방성 텐서.

    Returns:
        (η, ξ): η ≥ 0, ξ는 III_b 부호 따라.
    """
    II_b, III_b = invariants_II_III(b)
    eta = float(np.sqrt(max(II_b / 6.0, 0.0)))
    xi_cube = III_b / 6.0
    xi = float(np.sign(xi_cube) * np.cbrt(abs(xi_cube)))
    return eta, xi


def is_realizable(
    b: NDArray[np.float64],
) -> bool:
    """Lumley realizability 조건 확인 — η, ξ가 Lumley triangle 내부에 있는지.

    조건: 3-η + 9η² ≥ ξ³ ≥ 9η² - 3η (단순화 형태).

    Args:
        b: 비등방성 텐서.

    Returns:
        Realizability 여부.
    """
    eta, xi = lumley_eta_xi(b)
    # 단순화: η, |ξ| 모두 ≤ 1/3
    return eta <= 1.0 / 3.0 + 1e-6 and abs(xi) <= 1.0 / 3.0 + 1e-6


def turbulence_state(
    b: NDArray[np.float64],
) -> str:
    """Reynolds 응력 비등방성 상태 분류.

    Categories:
        - isotropic:        η ≈ 0
        - axisymmetric_expansion: ξ > 0, near disc/expansion side
        - axisymmetric_contraction: ξ < 0, rod-like
        - 2-component:      |ξ| 큰 영역
        - 1-component:      η ≈ 1/3

    Args:
        b: 비등방성 텐서.

    Returns:
        분류 문자열.
    """
    eta, xi = lumley_eta_xi(b)
    if eta < 0.05:
        return "isotropic"
    if eta > 0.30:
        return "1_component"
    if abs(xi) > 0.20:
        return "2_component"
    if xi > 0.05:
        return "axisymmetric_expansion"
    if xi < -0.05:
        return "axisymmetric_contraction"
    return "intermediate"


def reynolds_stress_eigenvalues(
    R: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Reynolds 응력 텐서 고유값 (오름차순).

    Args:
        R: (3, 3) 대칭.

    Returns:
        (3,) 고유값.
    """
    R = np.asarray(R, dtype=np.float64)
    if R.shape != (3, 3):
        raise ValueError(f"R must be (3, 3), got {R.shape}")
    R_sym = 0.5 * (R + R.T)
    eigvals = np.linalg.eigvalsh(R_sym)
    return np.sort(eigvals)


def barycentric_coordinates(
    b: NDArray[np.float64],
) -> tuple[float, float, float]:
    """Barycentric (componentality) 좌표 (C_1c, C_2c, C_3c).

    Banerjee et al. 2007: C_1c + C_2c + C_3c = 1.
    1c: 1-component, 2c: 2-component, 3c: 3-component (isotropic).

    Args:
        b: 비등방성 텐서.

    Returns:
        (C_1c, C_2c, C_3c).
    """
    b = np.asarray(b, dtype=np.float64)
    if b.shape != (3, 3):
        raise ValueError(f"b must be (3, 3), got {b.shape}")
    b_sym = 0.5 * (b + b.T)
    eigs = np.sort(np.linalg.eigvalsh(b_sym))[::-1]  # 내림차순
    lam1, lam2, lam3 = float(eigs[0]), float(eigs[1]), float(eigs[2])
    C_1c = lam1 - lam2
    C_2c = 2 * (lam2 - lam3)
    C_3c = 3 * lam3 + 1.0
    return C_1c, C_2c, C_3c


__all__ = [
    "anisotropy_tensor",
    "invariants_II_III",
    "lumley_eta_xi",
    "is_realizable",
    "turbulence_state",
    "reynolds_stress_eigenvalues",
    "barycentric_coordinates",
]
