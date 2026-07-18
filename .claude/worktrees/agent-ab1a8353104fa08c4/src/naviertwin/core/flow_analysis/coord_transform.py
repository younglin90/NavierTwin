"""좌표계 변환 — Cartesian ↔ Cylindrical ↔ Spherical.

CFD 결과 후처리에서 회전 대칭(터빈, 분수, 압축기)에 자주 사용. 위치 좌표
+ 벡터장(속도, 와도) 동시 변환 지원.

Convention:
    Cylindrical (r, θ, z):
        x = r cos θ, y = r sin θ, z = z
    Spherical (r, θ, φ):  ISO 80000-2 — θ = polar (z), φ = azimuth (x→y)
        x = r sin θ cos φ, y = r sin θ sin φ, z = r cos θ

Examples:
    >>> import numpy as np
    >>> xyz = np.array([[1.0, 0.0, 0.0]])
    >>> from naviertwin.core.flow_analysis.coord_transform import cart_to_cyl
    >>> r, theta, z = cart_to_cyl(xyz).T
    >>> abs(r[0] - 1.0) < 1e-12 and abs(theta[0]) < 1e-12
    True
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from naviertwin.utils.logger import get_logger

logger = get_logger(__name__)


def cart_to_cyl(xyz: NDArray[np.float64]) -> NDArray[np.float64]:
    """Cartesian → Cylindrical 위치 변환.

    Args:
        xyz: (N, 3) Cartesian 좌표.

    Returns:
        (N, 3) — (r, θ, z), θ ∈ [-π, π].

    Raises:
        ValueError: 형상 오류.
    """
    xyz = np.asarray(xyz, dtype=np.float64)
    if xyz.ndim != 2 or xyz.shape[1] != 3:
        raise ValueError(f"xyz shape must be (N, 3), got {xyz.shape}")
    r = np.sqrt(xyz[:, 0] ** 2 + xyz[:, 1] ** 2)
    theta = np.arctan2(xyz[:, 1], xyz[:, 0])
    z = xyz[:, 2]
    return np.column_stack([r, theta, z])


def cyl_to_cart(rtz: NDArray[np.float64]) -> NDArray[np.float64]:
    """Cylindrical → Cartesian 위치 변환.

    Args:
        rtz: (N, 3) (r, θ, z).

    Returns:
        (N, 3) Cartesian.
    """
    rtz = np.asarray(rtz, dtype=np.float64)
    if rtz.ndim != 2 or rtz.shape[1] != 3:
        raise ValueError(f"rtz shape must be (N, 3), got {rtz.shape}")
    r, theta, z = rtz.T
    x = r * np.cos(theta)
    y = r * np.sin(theta)
    return np.column_stack([x, y, z])


def cart_to_sph(xyz: NDArray[np.float64]) -> NDArray[np.float64]:
    """Cartesian → Spherical (r, θ, φ) ISO 80000-2.

    Args:
        xyz: (N, 3) Cartesian.

    Returns:
        (N, 3) (r, θ_polar, φ_azimuth). θ ∈ [0, π], φ ∈ [-π, π].
    """
    xyz = np.asarray(xyz, dtype=np.float64)
    if xyz.ndim != 2 or xyz.shape[1] != 3:
        raise ValueError(f"xyz shape must be (N, 3), got {xyz.shape}")
    r = np.linalg.norm(xyz, axis=1)
    theta = np.where(r > 1e-30, np.arccos(np.clip(xyz[:, 2] / np.maximum(r, 1e-30), -1, 1)), 0.0)
    phi = np.arctan2(xyz[:, 1], xyz[:, 0])
    return np.column_stack([r, theta, phi])


def sph_to_cart(rtp: NDArray[np.float64]) -> NDArray[np.float64]:
    """Spherical → Cartesian.

    Args:
        rtp: (N, 3) (r, θ_polar, φ_azimuth).

    Returns:
        (N, 3) Cartesian.
    """
    rtp = np.asarray(rtp, dtype=np.float64)
    if rtp.ndim != 2 or rtp.shape[1] != 3:
        raise ValueError(f"rtp shape must be (N, 3), got {rtp.shape}")
    r, theta, phi = rtp.T
    s = np.sin(theta)
    return np.column_stack([
        r * s * np.cos(phi),
        r * s * np.sin(phi),
        r * np.cos(theta),
    ])


def vector_cart_to_cyl(
    vec_xyz: NDArray[np.float64],
    pos_xyz: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Cartesian 벡터를 위치 의존 cylindrical (v_r, v_θ, v_z) 로 변환.

    각 점의 방위각 θ에 의해 회전 변환:
        v_r =  cos θ v_x + sin θ v_y
        v_θ = -sin θ v_x + cos θ v_y
        v_z =  v_z

    Args:
        vec_xyz: (N, 3) 벡터장.
        pos_xyz: (N, 3) 위치.

    Returns:
        (N, 3) 원통 좌표계 벡터.

    Raises:
        ValueError: 형상 불일치.
    """
    vec_xyz = np.asarray(vec_xyz, dtype=np.float64)
    pos_xyz = np.asarray(pos_xyz, dtype=np.float64)
    if vec_xyz.shape != pos_xyz.shape:
        raise ValueError(
            f"vec/pos shape mismatch: {vec_xyz.shape} vs {pos_xyz.shape}"
        )
    if vec_xyz.ndim != 2 or vec_xyz.shape[1] != 3:
        raise ValueError(f"shape must be (N, 3), got {vec_xyz.shape}")

    theta = np.arctan2(pos_xyz[:, 1], pos_xyz[:, 0])
    c, s = np.cos(theta), np.sin(theta)
    v_r = c * vec_xyz[:, 0] + s * vec_xyz[:, 1]
    v_t = -s * vec_xyz[:, 0] + c * vec_xyz[:, 1]
    v_z = vec_xyz[:, 2]
    return np.column_stack([v_r, v_t, v_z])


def vector_cyl_to_cart(
    vec_rtz: NDArray[np.float64],
    pos_xyz: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Cylindrical 벡터 → Cartesian.

    Args:
        vec_rtz: (N, 3) (v_r, v_θ, v_z).
        pos_xyz: (N, 3) 위치 (Cartesian).

    Returns:
        (N, 3) Cartesian 벡터.
    """
    vec_rtz = np.asarray(vec_rtz, dtype=np.float64)
    pos_xyz = np.asarray(pos_xyz, dtype=np.float64)
    if vec_rtz.shape != pos_xyz.shape or vec_rtz.shape[1] != 3:
        raise ValueError(
            f"shape mismatch or wrong dim: {vec_rtz.shape} vs {pos_xyz.shape}"
        )
    theta = np.arctan2(pos_xyz[:, 1], pos_xyz[:, 0])
    c, s = np.cos(theta), np.sin(theta)
    v_x = c * vec_rtz[:, 0] - s * vec_rtz[:, 1]
    v_y = s * vec_rtz[:, 0] + c * vec_rtz[:, 1]
    v_z = vec_rtz[:, 2]
    return np.column_stack([v_x, v_y, v_z])


def axis_align_rotation(
    axis: NDArray[np.float64],
    target: NDArray[np.float64] | None = None,
) -> NDArray[np.float64]:
    """주어진 axis를 target(기본 +z)에 정렬하는 회전 행렬을 반환한다.

    Rodrigues 공식 사용. 회전축 = axis × target.

    Args:
        axis: (3,) 정렬할 단위 벡터.
        target: (3,) 목표 방향 (기본 [0, 0, 1]).

    Returns:
        (3, 3) 회전 행렬 R, R @ axis ≈ target.

    Raises:
        ValueError: axis가 (3,)이 아닌 경우.
    """
    axis = np.asarray(axis, dtype=np.float64)
    if axis.shape != (3,):
        raise ValueError(f"axis must be (3,), got {axis.shape}")
    a = axis / max(np.linalg.norm(axis), 1e-30)
    if target is None:
        b = np.array([0.0, 0.0, 1.0])
    else:
        b = np.asarray(target, dtype=np.float64)
        b = b / max(np.linalg.norm(b), 1e-30)

    v = np.cross(a, b)
    s = np.linalg.norm(v)
    c = float(np.dot(a, b))
    if s < 1e-12:
        # 평행 또는 반대 방향
        if c > 0:
            return np.eye(3)
        # 180도 회전 — 임의 수직축 선택
        perp = np.array([1.0, 0.0, 0.0]) if abs(a[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
        v = np.cross(a, perp)
        v = v / max(np.linalg.norm(v), 1e-30)
        K = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
        return np.eye(3) + 2 * K @ K  # 180도

    K = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    R = np.eye(3) + K + K @ K * ((1 - c) / (s ** 2))
    return R


__all__ = [
    "cart_to_cyl",
    "cyl_to_cart",
    "cart_to_sph",
    "sph_to_cart",
    "vector_cart_to_cyl",
    "vector_cyl_to_cart",
    "axis_align_rotation",
]
