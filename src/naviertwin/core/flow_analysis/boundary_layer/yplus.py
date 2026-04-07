"""y+ 벽면 분석 모듈.

벽면 전단응력으로부터 마찰 속도(u_tau), 무차원 벽 거리(y+)를 계산하고
Schlichting 경계층 상관식으로 격자 첫 번째 셀 높이를 추정한다.

References:
    - Schlichting, H. & Gersten, K. (2016). Boundary-Layer Theory (10th ed.).
      Springer. (평판 난류 경계층 Cf 상관식)
    - NASA CFD validation guidelines. y+ estimation.

Examples:
    y+ 계산::

        import numpy as np
        from naviertwin.core.flow_analysis.boundary_layer.yplus import (
            compute_yplus,
            estimate_first_cell_height,
        )

        tau_w = np.array([[0.5, 0.0, 0.0], [0.4, 0.0, 0.0]])
        y_wall = np.array([1e-4, 1e-4])
        yp = compute_yplus(tau_w, rho=1.225, nu=1.5e-5, y_wall=y_wall)
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from naviertwin.utils.logger import get_logger

logger = get_logger(__name__)


def compute_yplus(
    wall_shear_stress: NDArray[np.float64],
    rho: float,
    nu: float,
    y_wall: NDArray[np.float64],
) -> NDArray[np.float64]:
    """y+ 를 계산한다.

    y+ = u_tau * y / nu

    여기서 u_tau 는 마찰 속도다.

    Args:
        wall_shear_stress: 벽면 전단응력 벡터, shape ``(N, 3)`` [Pa].
            각 행은 [tau_x, tau_y, tau_z] 성분이다.
        rho: 유체 밀도 [kg/m³].
        nu: 동점성계수 [m²/s].
        y_wall: 첫 번째 셀 중심까지 벽면으로부터의 거리 [m], shape ``(N,)``.

    Returns:
        무차원 벽 거리 y+, shape ``(N,)``.

    Raises:
        ValueError: ``rho`` 또는 ``nu`` 가 0 이하인 경우.

    Examples:
        >>> tau_w = np.array([[0.5, 0.0, 0.0]])
        >>> y_wall = np.array([1e-4])
        >>> yp = compute_yplus(tau_w, rho=1.225, nu=1.5e-5, y_wall=y_wall)
    """
    if rho <= 0:
        raise ValueError(f"밀도(rho)는 0 보다 커야 합니다. 입력값: {rho}")
    if nu <= 0:
        raise ValueError(
            f"동점성계수(nu)는 0 보다 커야 합니다. 입력값: {nu}"
        )

    wall_shear_stress = np.asarray(wall_shear_stress, dtype=np.float64)
    y_wall = np.asarray(y_wall, dtype=np.float64)

    u_tau = compute_friction_velocity(wall_shear_stress, rho)
    y_plus = u_tau * y_wall / nu

    logger.debug(
        "compute_yplus: y+ 범위 [%.3f, %.3f]",
        float(y_plus.min()),
        float(y_plus.max()),
    )
    return y_plus


def estimate_first_cell_height(
    y_plus_target: float,
    Re: float,
    L: float,
    nu: float,
    rho: float,
    U_inf: float,
) -> float:
    """Schlichting 평판 난류 경계층 상관식으로 첫 번째 셀 높이를 추정한다.

    적용 공식:
        Cf ≈ 0.026 × Re^(-1/7)    (Schlichting, 난류 평판)
        tau_w = Cf × 0.5 × rho × U_inf²
        u_tau = sqrt(tau_w / rho)
        y1 = y_plus_target × nu / u_tau

    Args:
        y_plus_target: 목표 y+ 값 (일반적으로 1~300).
        Re: 레이놀즈 수 (= U_inf × L / nu).
        L: 특성 길이 [m] (예: 평판 길이, 코드 길이).
        nu: 동점성계수 [m²/s].
        rho: 유체 밀도 [kg/m³].
        U_inf: 자유류 속도 [m/s].

    Returns:
        권장 첫 번째 셀 높이 y1 [m].

    Raises:
        ValueError: Re, U_inf, rho, nu 중 하나라도 0 이하인 경우.

    Examples:
        >>> y1 = estimate_first_cell_height(
        ...     y_plus_target=1.0,
        ...     Re=1e6, L=1.0, nu=1.5e-5, rho=1.225, U_inf=1.0
        ... )
    """
    for name, val in (("Re", Re), ("L", L), ("nu", nu), ("rho", rho), ("U_inf", U_inf)):
        if val <= 0:
            raise ValueError(f"{name} 은(는) 0 보다 커야 합니다. 입력값: {val}")

    # Schlichting 평판 난류 마찰 계수
    Cf = 0.026 * (Re ** (-1.0 / 7.0))
    tau_w = Cf * 0.5 * rho * U_inf**2
    u_tau = float(np.sqrt(tau_w / rho))

    if u_tau < 1e-16:
        raise ValueError(
            f"마찰 속도가 너무 작습니다 (u_tau={u_tau:.3e}). "
            "입력 파라미터를 확인하세요."
        )

    y1 = y_plus_target * nu / u_tau

    logger.debug(
        "estimate_first_cell_height: Cf=%.4e, tau_w=%.4e, u_tau=%.4e, y1=%.4e m",
        Cf,
        tau_w,
        u_tau,
        y1,
    )
    return y1


def compute_friction_velocity(
    wall_shear_stress: NDArray[np.float64],
    rho: float,
) -> NDArray[np.float64]:
    """마찰 속도 u_tau 를 계산한다.

    u_tau = sqrt(|tau_w| / rho)

    여기서 |tau_w| 는 전단응력 벡터의 크기다.

    Args:
        wall_shear_stress: 벽면 전단응력 벡터, shape ``(N, 3)`` [Pa].
        rho: 유체 밀도 [kg/m³].

    Returns:
        마찰 속도 u_tau [m/s], shape ``(N,)``.

    Raises:
        ValueError: ``rho`` 가 0 이하인 경우.

    Examples:
        >>> tau_w = np.array([[0.5, 0.0, 0.0], [0.3, 0.0, 0.0]])
        >>> u_tau = compute_friction_velocity(tau_w, rho=1.225)
    """
    if rho <= 0:
        raise ValueError(f"밀도(rho)는 0 보다 커야 합니다. 입력값: {rho}")

    wall_shear_stress = np.asarray(wall_shear_stress, dtype=np.float64)

    if wall_shear_stress.ndim == 1:
        # 단일 벡터 처리
        tau_mag = np.linalg.norm(wall_shear_stress)
        u_tau = np.sqrt(tau_mag / rho)
        return np.array([u_tau], dtype=np.float64)

    # shape (N, 3) → (N,)
    tau_mag = np.linalg.norm(wall_shear_stress, axis=-1)
    u_tau = np.sqrt(tau_mag / rho)

    logger.debug(
        "compute_friction_velocity: u_tau 범위 [%.4e, %.4e] m/s",
        float(u_tau.min()),
        float(u_tau.max()),
    )
    return u_tau


def compute_wall_units(
    y_wall: NDArray[np.float64],
    u_tau: NDArray[np.float64],
    nu: float,
) -> dict[str, NDArray[np.float64]]:
    """벽 단위(y+, 점성 길이 스케일 delta_nu)를 계산한다.

    y+      = u_tau * y / nu
    delta_nu = nu / u_tau   (점성 길이 스케일)

    Args:
        y_wall: 벽면으로부터 첫 번째 셀 중심까지의 거리 [m], shape ``(N,)``.
        u_tau: 마찰 속도 [m/s], shape ``(N,)``.
        nu: 동점성계수 [m²/s].

    Returns:
        딕셔너리:
            - ``"y_plus"``: 무차원 벽 거리, shape ``(N,)``
            - ``"delta_nu"``: 점성 길이 스케일 [m], shape ``(N,)``

    Raises:
        ValueError: ``nu`` 가 0 이하인 경우.

    Examples:
        >>> y_plus_dict = compute_wall_units(y_wall, u_tau, nu=1.5e-5)
    """
    if nu <= 0:
        raise ValueError(
            f"동점성계수(nu)는 0 보다 커야 합니다. 입력값: {nu}"
        )

    y_wall = np.asarray(y_wall, dtype=np.float64)
    u_tau = np.asarray(u_tau, dtype=np.float64)

    # u_tau 가 0 인 점은 delta_nu = 0 처리 (벽면 속도 = 0 부동소수 방지)
    with np.errstate(divide="ignore", invalid="ignore"):
        delta_nu = np.where(u_tau > 0.0, nu / u_tau, 0.0)
        y_plus = np.where(u_tau > 0.0, u_tau * y_wall / nu, 0.0)

    logger.debug(
        "compute_wall_units: y+ 범위 [%.3f, %.3f], "
        "delta_nu 범위 [%.3e, %.3e] m",
        float(y_plus.min()),
        float(y_plus.max()),
        float(delta_nu.min()),
        float(delta_nu.max()),
    )

    return {
        "y_plus": y_plus.astype(np.float64),
        "delta_nu": delta_nu.astype(np.float64),
    }
