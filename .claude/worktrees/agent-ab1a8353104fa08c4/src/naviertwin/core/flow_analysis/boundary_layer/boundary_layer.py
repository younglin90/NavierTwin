"""경계층 두께 및 마찰계수 계산.

1D 수직 속도 프로파일 u(y) 로부터:
    - δ₉₉ : 99% 경계층 두께
    - δ* : displacement thickness ∫₀^∞ (1 - u/U∞) dy
    - θ  : momentum thickness ∫₀^∞ (u/U∞)(1 - u/U∞) dy
    - H  : shape factor δ*/θ
    - Cf : skin friction coefficient 2·τ_w / (ρ·U∞²)

Examples:
    >>> import numpy as np
    >>> from naviertwin.core.flow_analysis.boundary_layer.boundary_layer import (
    ...     boundary_layer_thicknesses, skin_friction,
    ... )
    >>> y = np.linspace(0, 0.05, 200)
    >>> U_inf = 10.0
    >>> u = U_inf * (1 - np.exp(-y / 0.005))  # 지수형 프로파일
    >>> thick = boundary_layer_thicknesses(y, u, U_inf)
    >>> Cf = skin_friction(tau_w=0.5, rho=1.0, U_inf=U_inf)
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from naviertwin.utils.logger import get_logger

logger = get_logger(__name__)


def boundary_layer_thicknesses(
    y: NDArray[np.float64],
    u: NDArray[np.float64],
    U_inf: float,
) -> dict[str, float]:
    """δ₉₉, δ*, θ, H 를 계산한다.

    Args:
        y: 벽면 수직 좌표 (오름차순). shape = (n,).
        u: 속도 프로파일. shape = (n,).
        U_inf: 자유류 속도.

    Returns:
        {"delta99": ..., "delta_star": ..., "theta": ..., "H": ...}

    Raises:
        ValueError: 크기가 맞지 않거나 U_inf <= 0 인 경우.
    """
    y = np.asarray(y, dtype=np.float64)
    u = np.asarray(u, dtype=np.float64)
    if y.shape != u.shape:
        raise ValueError(
            f"y 와 u 의 shape 가 달라요: {y.shape} vs {u.shape}"
        )
    if U_inf <= 0:
        raise ValueError(f"U_inf 는 양수여야 합니다: {U_inf}")

    ratio = np.clip(u / U_inf, 0.0, 1.0)

    # δ99 — ratio 가 0.99 이상이 되는 첫 y
    idx = np.argmax(ratio >= 0.99)
    if ratio[idx] >= 0.99:
        delta99 = float(y[idx])
    else:
        delta99 = float(y[-1])

    delta_star = float(np.trapezoid(1.0 - ratio, y))
    theta = float(np.trapezoid(ratio * (1.0 - ratio), y))
    H = delta_star / theta if theta > 0 else float("inf")

    logger.debug(
        "BL: δ99=%.4g, δ*=%.4g, θ=%.4g, H=%.4g", delta99, delta_star, theta, H
    )
    return {
        "delta99": delta99,
        "delta_star": delta_star,
        "theta": theta,
        "H": H,
    }


def skin_friction(tau_w: float, rho: float, U_inf: float) -> float:
    """Cf = 2·τ_w / (ρ·U∞²).

    Args:
        tau_w: 벽면 전단응력 [Pa].
        rho: 밀도 [kg/m³].
        U_inf: 자유류 속도.

    Returns:
        마찰계수 Cf (무차원).
    """
    if U_inf <= 0 or rho <= 0:
        raise ValueError(f"U_inf 와 rho 는 양수여야 합니다: rho={rho}, U_inf={U_inf}")
    return float(2.0 * tau_w / (rho * U_inf**2))


__all__ = ["boundary_layer_thicknesses", "skin_friction"]
