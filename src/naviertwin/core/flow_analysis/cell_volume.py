"""비정규 격자 셀 볼륨 + 가중 체적 적분.

3D 사면체(tet)/육면체(hex)/삼각프리즘(prism)/피라미드 등 셀 볼륨 계산.
체적 가중 평균/적분 (난류 통계 정확도에 필수).

상용 툴 대응:
    - Tecplot 360: Compute Cell Volume + Volume Integral
    - Ansys CFD-Post: Cell Volumes (mesh statistics)
    - EnSight: Variable Calculator (Volume)

Examples:
    >>> import numpy as np
    >>> # 단위 사면체 (V = 1/6)
    >>> verts = np.array([[0,0,0], [1,0,0], [0,1,0], [0,0,1]], dtype=float)
    >>> from naviertwin.core.flow_analysis.cell_volume import tet_volume
    >>> abs(tet_volume(verts) - 1/6) < 1e-12
    True
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from naviertwin.utils.logger import get_logger

logger = get_logger(__name__)


def tet_volume(vertices: NDArray[np.float64]) -> float:
    """사면체 볼륨 V = |det([v1-v0, v2-v0, v3-v0])| / 6.

    Args:
        vertices: (4, 3) 4개 정점 좌표.

    Returns:
        체적 (양수).

    Raises:
        ValueError: 형상 오류.
    """
    v = np.asarray(vertices, dtype=np.float64)
    if v.shape != (4, 3):
        raise ValueError(f"vertices must be (4, 3), got {v.shape}")
    a = v[1] - v[0]
    b = v[2] - v[0]
    c = v[3] - v[0]
    return float(abs(np.dot(a, np.cross(b, c))) / 6.0)


def tet_volumes_batch(
    vertices: NDArray[np.float64],
    connectivity: NDArray[np.intp],
) -> NDArray[np.float64]:
    """배치 사면체 볼륨.

    Args:
        vertices: (N, 3) 점 좌표.
        connectivity: (n_cells, 4) 사면체 셀 정점 인덱스.

    Returns:
        (n_cells,) 체적 배열.

    Raises:
        ValueError: 형상 오류.
    """
    V = np.asarray(vertices, dtype=np.float64)
    C = np.asarray(connectivity, dtype=np.intp)
    if V.ndim != 2 or V.shape[1] != 3:
        raise ValueError(f"vertices shape (N, 3), got {V.shape}")
    if C.ndim != 2 or C.shape[1] != 4:
        raise ValueError(f"connectivity (n_cells, 4), got {C.shape}")

    p0 = V[C[:, 0]]
    p1 = V[C[:, 1]]
    p2 = V[C[:, 2]]
    p3 = V[C[:, 3]]
    a = p1 - p0
    b = p2 - p0
    c = p3 - p0
    cross = np.cross(b, c)
    triple = np.einsum("ij,ij->i", a, cross)
    return np.abs(triple) / 6.0


def hex_volume(vertices: NDArray[np.float64]) -> float:
    """육면체 볼륨 — 5-사면체 분할법.

    표준 vertex 순서 (VTK_HEXAHEDRON):
        0: (0,0,0), 1: (1,0,0), 2: (1,1,0), 3: (0,1,0)
        4: (0,0,1), 5: (1,0,1), 6: (1,1,1), 7: (0,1,1)

    Args:
        vertices: (8, 3) 정점.

    Returns:
        체적.

    Raises:
        ValueError: 형상 오류.
    """
    v = np.asarray(vertices, dtype=np.float64)
    if v.shape != (8, 3):
        raise ValueError(f"vertices must be (8, 3), got {v.shape}")

    # 5-tet decomposition
    tets = [
        [0, 1, 3, 4],
        [1, 2, 3, 6],
        [1, 3, 4, 6],
        [3, 4, 6, 7],
        [1, 4, 5, 6],
    ]
    return sum(tet_volume(v[tet]) for tet in tets)


def pyramid_volume(vertices: NDArray[np.float64]) -> float:
    """사각뿔 볼륨 V = (1/3) · A_base · h.

    실제로는 2-사면체 분할.

    정점 순서: 0,1,2,3 = base (반시계), 4 = apex.

    Args:
        vertices: (5, 3).

    Returns:
        체적.
    """
    v = np.asarray(vertices, dtype=np.float64)
    if v.shape != (5, 3):
        raise ValueError(f"vertices must be (5, 3), got {v.shape}")
    # 2 tets
    return tet_volume(v[[0, 1, 2, 4]]) + tet_volume(v[[0, 2, 3, 4]])


def prism_volume(vertices: NDArray[np.float64]) -> float:
    """삼각 프리즘 (wedge) 볼륨 — 3-사면체 분할.

    정점 순서: 0,1,2 = 아래 삼각형, 3,4,5 = 위 삼각형 (대응).

    Args:
        vertices: (6, 3).

    Returns:
        체적.
    """
    v = np.asarray(vertices, dtype=np.float64)
    if v.shape != (6, 3):
        raise ValueError(f"vertices must be (6, 3), got {v.shape}")
    return (
        tet_volume(v[[0, 1, 2, 3]])
        + tet_volume(v[[1, 2, 3, 4]])
        + tet_volume(v[[2, 3, 4, 5]])
    )


def volume_integral(
    cell_volumes: NDArray[np.float64],
    cell_field: NDArray[np.float64],
) -> float | NDArray[np.float64]:
    """∫ φ dV = Σ φ_i V_i.

    Args:
        cell_volumes: (n_cells,) 셀 체적.
        cell_field: (n_cells,) 또는 (n_cells, k) 셀 중심 값.

    Returns:
        스칼라 또는 (k,) 적분값.

    Raises:
        ValueError: 형상 불일치.
    """
    V = np.asarray(cell_volumes, dtype=np.float64)
    f = np.asarray(cell_field, dtype=np.float64)
    if V.ndim != 1:
        raise ValueError(f"cell_volumes must be 1D, got {V.shape}")
    if f.shape[0] != V.shape[0]:
        raise ValueError(
            f"cell_field length {f.shape[0]} != cells {V.shape[0]}"
        )

    if f.ndim == 1:
        return float(np.sum(V * f))
    return np.sum(V[:, None] * f, axis=0)


def volume_average(
    cell_volumes: NDArray[np.float64],
    cell_field: NDArray[np.float64],
) -> float | NDArray[np.float64]:
    """⟨φ⟩_V = ∫ φ dV / ∫ dV.

    Args:
        cell_volumes: 셀 체적.
        cell_field: 셀 중심 값.

    Returns:
        체적 가중 평균.
    """
    V = np.asarray(cell_volumes, dtype=np.float64)
    total_V = float(V.sum())
    if total_V < 1e-30:
        f = np.asarray(cell_field, dtype=np.float64)
        return 0.0 if f.ndim == 1 else np.zeros(f.shape[1:])
    return volume_integral(cell_volumes, cell_field) / total_V


def volume_weighted_variance(
    cell_volumes: NDArray[np.float64],
    cell_field: NDArray[np.float64],
) -> float:
    """체적 가중 분산 σ² = ⟨(φ - ⟨φ⟩)²⟩_V.

    Args:
        cell_volumes: 셀 체적.
        cell_field: 셀 값.

    Returns:
        분산.
    """
    avg = volume_average(cell_volumes, cell_field)
    deviation = np.asarray(cell_field) - avg
    return float(volume_average(cell_volumes, deviation ** 2))


def cell_centroids(
    vertices: NDArray[np.float64],
    connectivity: NDArray[np.intp],
) -> NDArray[np.float64]:
    """각 셀 중심점 = 정점들의 평균.

    Args:
        vertices: (N, 3) 점.
        connectivity: (n_cells, k) 셀당 정점 인덱스 (k=4: tet, 8: hex 등).

    Returns:
        (n_cells, 3) 중심점.

    Raises:
        ValueError: 형상 오류.
    """
    V = np.asarray(vertices, dtype=np.float64)
    C = np.asarray(connectivity, dtype=np.intp)
    if V.ndim != 2 or V.shape[1] != 3:
        raise ValueError(f"vertices shape (N, 3), got {V.shape}")
    if C.ndim != 2:
        raise ValueError(f"connectivity must be 2D, got {C.shape}")

    cell_pts = V[C]  # (n_cells, k, 3)
    return cell_pts.mean(axis=1)


__all__ = [
    "tet_volume",
    "tet_volumes_batch",
    "hex_volume",
    "pyramid_volume",
    "prism_volume",
    "volume_integral",
    "volume_average",
    "volume_weighted_variance",
    "cell_centroids",
]
