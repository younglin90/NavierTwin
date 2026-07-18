"""평면 통과 플럭스 적분 — 질량/운동량/에너지/스칼라 플럭스.

CFD 결과의 임의 평면을 통과하는 보존량 플럭스를 계산한다.
상용 후처리 툴의 핵심 기능 (CFD-Post Surface Integral, EnSight Flux Calc).

Examples:
    >>> import numpy as np
    >>> # 단위 면적 (z=0 평면, normal=+z)
    >>> tris = np.array([[[0,0,0],[1,0,0],[1,1,0]],
    ...                  [[0,0,0],[1,1,0],[0,1,0]]])
    >>> u = np.array([[0,0,1.0],[0,0,1.0]])  # +z 방향 1 m/s
    >>> rho = np.array([1.0, 1.0])
    >>> from naviertwin.core.flow_analysis.plane_flux import mass_flux
    >>> abs(mass_flux(tris, u, rho) - 1.0) < 1e-10
    True
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from naviertwin.core.flow_analysis.surface_integrals import triangle_normal_area
from naviertwin.utils.logger import get_logger

logger = get_logger(__name__)


def mass_flux(
    triangles: NDArray[np.float64],
    velocity: NDArray[np.float64],
    density: NDArray[np.float64] | float = 1.0,
) -> float:
    """질량 플럭스 ṁ = ∫ ρ (u · n) dA.

    Args:
        triangles: (n_faces, 3, 3) 삼각형 정점.
        velocity: (n_faces, 3) 면당 평균 속도 벡터.
        density: (n_faces,) 또는 스칼라 밀도.

    Returns:
        총 질량 플럭스 (양수: 외향).

    Raises:
        ValueError: 형상 불일치.
    """
    velocity = np.asarray(velocity, dtype=np.float64)
    n_hat, A = triangle_normal_area(triangles)
    if velocity.shape != (n_hat.shape[0], 3):
        raise ValueError(
            f"velocity shape {velocity.shape} != ({n_hat.shape[0]}, 3)"
        )

    rho = np.asarray(density, dtype=np.float64)
    if rho.ndim == 0:
        rho = np.full(n_hat.shape[0], float(rho))
    if rho.shape != (n_hat.shape[0],):
        raise ValueError(
            f"density shape {rho.shape} != ({n_hat.shape[0]},)"
        )

    u_dot_n = np.sum(velocity * n_hat, axis=1)
    return float(np.sum(rho * u_dot_n * A))


def volumetric_flow_rate(
    triangles: NDArray[np.float64],
    velocity: NDArray[np.float64],
) -> float:
    """체적 유량 Q = ∫ (u · n) dA.

    Args:
        triangles: (n_faces, 3, 3) 평면 삼각형.
        velocity: (n_faces, 3) 평균 속도.

    Returns:
        Q [m³/s].
    """
    return mass_flux(triangles, velocity, density=1.0)


def momentum_flux(
    triangles: NDArray[np.float64],
    velocity: NDArray[np.float64],
    density: NDArray[np.float64] | float = 1.0,
) -> NDArray[np.float64]:
    """운동량 플럭스 벡터 ∫ ρ u (u · n) dA.

    Args:
        triangles: (n_faces, 3, 3).
        velocity: (n_faces, 3).
        density: 밀도.

    Returns:
        (3,) 운동량 플럭스 벡터.
    """
    velocity = np.asarray(velocity, dtype=np.float64)
    n_hat, A = triangle_normal_area(triangles)
    if velocity.shape != (n_hat.shape[0], 3):
        raise ValueError(
            f"velocity shape {velocity.shape} != ({n_hat.shape[0]}, 3)"
        )

    rho = np.asarray(density, dtype=np.float64)
    if rho.ndim == 0:
        rho = np.full(n_hat.shape[0], float(rho))

    u_dot_n = np.sum(velocity * n_hat, axis=1)
    weights = rho * u_dot_n * A
    return np.sum(velocity * weights[:, None], axis=0)


def scalar_flux(
    triangles: NDArray[np.float64],
    velocity: NDArray[np.float64],
    scalar: NDArray[np.float64],
    density: NDArray[np.float64] | float = 1.0,
) -> float:
    """일반 스칼라 (예: 온도, 종 농도) 플럭스 ∫ ρ φ (u · n) dA.

    Args:
        triangles: (n_faces, 3, 3).
        velocity: (n_faces, 3).
        scalar: (n_faces,) 면당 스칼라 값.
        density: 밀도.

    Returns:
        스칼라 플럭스.

    Raises:
        ValueError: scalar 형상 불일치.
    """
    velocity = np.asarray(velocity, dtype=np.float64)
    scalar = np.asarray(scalar, dtype=np.float64)
    n_hat, A = triangle_normal_area(triangles)
    if scalar.shape != (n_hat.shape[0],):
        raise ValueError(
            f"scalar shape {scalar.shape} != ({n_hat.shape[0]},)"
        )

    rho = np.asarray(density, dtype=np.float64)
    if rho.ndim == 0:
        rho = np.full(n_hat.shape[0], float(rho))

    u_dot_n = np.sum(velocity * n_hat, axis=1)
    return float(np.sum(rho * scalar * u_dot_n * A))


def kinetic_energy_flux(
    triangles: NDArray[np.float64],
    velocity: NDArray[np.float64],
    density: NDArray[np.float64] | float = 1.0,
) -> float:
    """운동 에너지 플럭스 ∫ ρ ½|u|² (u · n) dA.

    Args:
        triangles: (n_faces, 3, 3).
        velocity: (n_faces, 3).
        density: 밀도.

    Returns:
        KE 플럭스.
    """
    velocity = np.asarray(velocity, dtype=np.float64)
    speed_sq = np.sum(velocity * velocity, axis=1)
    return scalar_flux(triangles, velocity, 0.5 * speed_sq, density)


def area_average(
    triangles: NDArray[np.float64],
    field: NDArray[np.float64],
) -> NDArray[np.float64] | float:
    """면적 가중 평균 ⟨φ⟩ = ∫ φ dA / ∫ dA.

    Args:
        triangles: (n_faces, 3, 3).
        field: (n_faces,) 또는 (n_faces, k) 면당 값.

    Returns:
        스칼라 또는 (k,) 평균.

    Raises:
        ValueError: field 형상 불일치.
    """
    field = np.asarray(field, dtype=np.float64)
    _, A = triangle_normal_area(triangles)
    total_area = float(A.sum())
    if total_area < 1e-30:
        return 0.0 if field.ndim == 1 else np.zeros(field.shape[1:])

    if field.ndim == 1:
        if field.shape[0] != A.shape[0]:
            raise ValueError(
                f"field shape {field.shape} != ({A.shape[0]},)"
            )
        return float(np.sum(field * A) / total_area)

    if field.shape[0] != A.shape[0]:
        raise ValueError(
            f"field shape {field.shape} != ({A.shape[0]}, ...)"
        )
    return np.sum(field * A[:, None], axis=0) / total_area


def mass_weighted_average(
    triangles: NDArray[np.float64],
    velocity: NDArray[np.float64],
    field: NDArray[np.float64],
    density: NDArray[np.float64] | float = 1.0,
) -> float:
    """질량 가중 평균 ⟨φ⟩_m = ∫ ρ φ (u · n) dA / ∫ ρ (u · n) dA.

    Bulk 온도, 평균 속도 등 계산에 사용.

    Args:
        triangles, velocity: 평면 + 속도.
        field: (n_faces,) 가중되는 스칼라.
        density: 밀도.

    Returns:
        질량 가중 평균.
    """
    numerator = scalar_flux(triangles, velocity, field, density)
    denominator = mass_flux(triangles, velocity, density)
    if abs(denominator) < 1e-30:
        return 0.0
    return numerator / denominator


__all__ = [
    "mass_flux",
    "volumetric_flow_rate",
    "momentum_flux",
    "scalar_flux",
    "kinetic_energy_flux",
    "area_average",
    "mass_weighted_average",
]
