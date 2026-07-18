"""표면 적분 — 압력/전단응력에서 힘·모멘트 계산.

Tecplot 360, CFD-Post, EnSight의 핵심 후처리 기능. 익형/차체 등
표면 패치에서 lift/drag/moment 적분을 수행한다.

Coefficient 정의:
    C_L = F_y / (½ρ U_∞² A_ref)
    C_D = F_x / (½ρ U_∞² A_ref)
    C_M = M_z / (½ρ U_∞² A_ref · L_ref)

Examples:
    >>> import numpy as np
    >>> # 단위 사각 표면, 균일 압력 P=1
    >>> faces = np.array([[[0,0,0], [1,0,0], [1,1,0]],
    ...                   [[0,0,0], [1,1,0], [0,1,0]]])
    >>> P = np.array([1.0, 1.0])
    >>> from naviertwin.core.flow_analysis.surface_integrals import (
    ...     pressure_force, force_coefficient
    ... )
    >>> F = pressure_force(faces, P)
    >>> abs(F[2] + 1.0) < 1e-10  # F_z = -1 (P 적용은 -n 방향)
    True
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from naviertwin.utils.logger import get_logger

logger = get_logger(__name__)


def triangle_normal_area(
    triangles: NDArray[np.float64],
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """삼각형 (n_faces, 3, 3) → (normals, areas).

    Args:
        triangles: (n_faces, 3, 3) 정점 좌표.

    Returns:
        (normals: (n_faces, 3) 단위 외법선,
         areas:   (n_faces,)  면적).

    Raises:
        ValueError: 형상 오류.
    """
    triangles = np.asarray(triangles, dtype=np.float64)
    if triangles.ndim != 3 or triangles.shape[1:] != (3, 3):
        raise ValueError(
            f"triangles shape must be (n_faces, 3, 3), got {triangles.shape}"
        )

    p0, p1, p2 = triangles[:, 0], triangles[:, 1], triangles[:, 2]
    cross = np.cross(p1 - p0, p2 - p0)
    areas = 0.5 * np.linalg.norm(cross, axis=1)
    norms = cross / np.maximum(2.0 * areas[:, None], 1e-30)
    return norms, areas


def pressure_force(
    triangles: NDArray[np.float64],
    pressure: NDArray[np.float64],
    reference_pressure: float = 0.0,
) -> NDArray[np.float64]:
    """표면에 작용하는 총 압력 힘 F = -∫(p - p_ref) n dA.

    부호 컨벤션: 외법선 n은 표면 바깥 방향, F는 유체가 표면에 가하는 힘.

    Args:
        triangles: (n_faces, 3, 3) 면 정점 좌표.
        pressure: (n_faces,) 면 중심 또는 면 평균 압력.
        reference_pressure: 기준 압력 (기본 0). 차압 적분.

    Returns:
        (3,) 총 힘 벡터 [F_x, F_y, F_z].

    Raises:
        ValueError: 형상 불일치.
    """
    pressure = np.asarray(pressure, dtype=np.float64)
    n_hat, A = triangle_normal_area(triangles)
    if pressure.shape != (n_hat.shape[0],):
        raise ValueError(
            f"pressure shape {pressure.shape} != ({n_hat.shape[0]},)"
        )

    dp = pressure - reference_pressure
    # F = -∫ p n dA  (유체가 surface에 미는 힘)
    F = -np.sum((dp * A)[:, None] * n_hat, axis=0)
    return F


def viscous_force(
    triangles: NDArray[np.float64],
    shear_traction: NDArray[np.float64],
) -> NDArray[np.float64]:
    """전단 응력 벡터에 의한 힘 F = ∫ τ_w dA.

    Args:
        triangles: (n_faces, 3, 3) 면 정점 좌표.
        shear_traction: (n_faces, 3) 면당 전단응력 벡터 (벽 접선 방향).

    Returns:
        (3,) 점성 힘.

    Raises:
        ValueError: 형상 불일치.
    """
    shear_traction = np.asarray(shear_traction, dtype=np.float64)
    _, A = triangle_normal_area(triangles)
    if shear_traction.shape != (A.shape[0], 3):
        raise ValueError(
            f"shear_traction shape {shear_traction.shape} != ({A.shape[0]}, 3)"
        )
    return np.sum(shear_traction * A[:, None], axis=0)


def total_force(
    triangles: NDArray[np.float64],
    pressure: NDArray[np.float64],
    shear_traction: NDArray[np.float64] | None = None,
    reference_pressure: float = 0.0,
) -> NDArray[np.float64]:
    """압력 + 점성 총 힘.

    Args:
        triangles: (n_faces, 3, 3) 면.
        pressure: (n_faces,) 압력.
        shear_traction: (n_faces, 3) 전단 응력 (선택).
        reference_pressure: 기준 압력.

    Returns:
        (3,) 총 힘.
    """
    F = pressure_force(triangles, pressure, reference_pressure)
    if shear_traction is not None:
        F = F + viscous_force(triangles, shear_traction)
    return F


def moment_about(
    triangles: NDArray[np.float64],
    pressure: NDArray[np.float64],
    center: NDArray[np.float64],
    shear_traction: NDArray[np.float64] | None = None,
    reference_pressure: float = 0.0,
) -> NDArray[np.float64]:
    """기준점 c에 대한 모멘트 M = ∫ (r - c) × dF.

    Args:
        triangles: (n_faces, 3, 3) 면.
        pressure: (n_faces,) 압력.
        center: (3,) 모멘트 기준점.
        shear_traction: (n_faces, 3) 전단 (선택).
        reference_pressure: 기준 압력.

    Returns:
        (3,) 모멘트 벡터.
    """
    triangles = np.asarray(triangles, dtype=np.float64)
    pressure = np.asarray(pressure, dtype=np.float64)
    center = np.asarray(center, dtype=np.float64)

    n_hat, A = triangle_normal_area(triangles)
    centroid = triangles.mean(axis=1)  # (n_faces, 3)
    r = centroid - center

    dp = pressure - reference_pressure
    dF = -((dp * A)[:, None] * n_hat)  # (n_faces, 3) 압력 힘 dF
    M = np.sum(np.cross(r, dF), axis=0)

    if shear_traction is not None:
        shear = np.asarray(shear_traction, dtype=np.float64)
        dF_v = shear * A[:, None]
        M = M + np.sum(np.cross(r, dF_v), axis=0)

    return M


def force_coefficient(
    F: float | NDArray[np.float64],
    rho: float,
    U_inf: float,
    A_ref: float,
) -> float | NDArray[np.float64]:
    """C_F = F / (½ρU_∞² A_ref).

    Args:
        F: 힘 (스칼라 또는 벡터).
        rho: 유체 밀도.
        U_inf: 자유흐름 속도.
        A_ref: 기준 면적.

    Returns:
        무차원 힘 계수.

    Raises:
        ValueError: rho 또는 U_inf 또는 A_ref가 양수가 아닌 경우.
    """
    if rho <= 0 or U_inf <= 0 or A_ref <= 0:
        raise ValueError(
            f"rho/U_inf/A_ref must be > 0, got {rho}, {U_inf}, {A_ref}"
        )
    q_inf = 0.5 * rho * U_inf * U_inf
    return F / (q_inf * A_ref)


def moment_coefficient(
    M: float | NDArray[np.float64],
    rho: float,
    U_inf: float,
    A_ref: float,
    L_ref: float,
) -> float | NDArray[np.float64]:
    """C_M = M / (½ρU_∞² A_ref L_ref).

    Args:
        M: 모멘트.
        rho, U_inf, A_ref: force_coefficient와 동일.
        L_ref: 기준 길이 (예: chord, diameter).

    Returns:
        무차원 모멘트 계수.

    Raises:
        ValueError: 입력이 양수가 아닌 경우.
    """
    if rho <= 0 or U_inf <= 0 or A_ref <= 0 or L_ref <= 0:
        raise ValueError(
            f"all positive required, got {rho}, {U_inf}, {A_ref}, {L_ref}"
        )
    q_inf = 0.5 * rho * U_inf * U_inf
    return M / (q_inf * A_ref * L_ref)


def lift_drag_split(
    F: NDArray[np.float64],
    flow_direction: NDArray[np.float64],
    lift_direction: NDArray[np.float64] | None = None,
) -> tuple[float, float]:
    """힘 벡터를 lift / drag로 분해.

    flow_direction(자유흐름 방향)과 lift_direction(수직 방향)을 받아
    drag = F · û_∞, lift = F · ŷ_lift.

    Args:
        F: (3,) 또는 (2,) 힘 벡터.
        flow_direction: 자유흐름 방향 단위벡터.
        lift_direction: lift 방향 단위벡터. None이면 흐름과 수직 방향 자동 선택.

    Returns:
        Tuple[lift, drag]: 스칼라.

    Raises:
        ValueError: 차원 불일치.
    """
    F = np.asarray(F, dtype=np.float64)
    flow = np.asarray(flow_direction, dtype=np.float64)
    if F.shape != flow.shape:
        raise ValueError(
            f"F and flow_direction shape mismatch: {F.shape} vs {flow.shape}"
        )

    flow_norm = flow / max(np.linalg.norm(flow), 1e-30)
    drag = float(np.dot(F, flow_norm))

    if lift_direction is not None:
        lift_dir = np.asarray(lift_direction, dtype=np.float64)
        lift_dir = lift_dir / max(np.linalg.norm(lift_dir), 1e-30)
    else:
        # 자동: 흐름 성분 제거
        F_perp = F - drag * flow_norm
        F_perp_norm = np.linalg.norm(F_perp)
        if F_perp_norm < 1e-12:
            return 0.0, drag
        lift_dir = F_perp / F_perp_norm

    lift = float(np.dot(F, lift_dir))
    return lift, drag


__all__ = [
    "triangle_normal_area",
    "pressure_force",
    "viscous_force",
    "total_force",
    "moment_about",
    "force_coefficient",
    "moment_coefficient",
    "lift_drag_split",
]
