"""pytest 공통 픽스처 모음.

pyvista 없이도 실행 가능하도록 numpy만 사용하여 가짜 메쉬 객체를 구성한다.
실제 PyVista 연동 테스트는 ``pytest.importorskip("pyvista")`` 로 처리한다.

Usage:
    모든 테스트에서 ``sample_mesh`` 픽스처를 자동으로 사용 가능.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest


class _FakeMeshData(dict):
    """point_data / cell_data 역할을 하는 딕셔너리 서브클래스."""


class _FakeUnstructuredGrid:
    """PyVista UnstructuredGrid의 최소 호환 가짜 객체.

    pyvista 없이도 테스트할 수 있도록 NumPy 배열로 구성된다.
    실제 PyVista 메서드 시그니처와 호환되도록 설계한다.

    Attributes:
        points: 노드 좌표 행렬. shape = (n_points, 3).
        cells: 셀 연결 정보 배열 (VTK 형식).
        cell_types: 셀 타입 배열.
        point_data: 점 데이터 딕셔너리.
        cell_data: 셀 데이터 딕셔너리.
    """

    def __init__(
        self,
        points: np.ndarray,
        cells: np.ndarray,
        cell_types: np.ndarray,
    ) -> None:
        self.points: np.ndarray = points
        self.cells: np.ndarray = cells
        self.cell_types: np.ndarray = cell_types
        self.point_data: _FakeMeshData = _FakeMeshData()
        self.cell_data: _FakeMeshData = _FakeMeshData()

    @property
    def n_points(self) -> int:
        """점 개수를 반환한다."""
        return len(self.points)

    @property
    def n_cells(self) -> int:
        """셀 개수를 반환한다."""
        return len(self.cell_types)

    def __repr__(self) -> str:
        return (
            f"_FakeUnstructuredGrid("
            f"n_points={self.n_points}, n_cells={self.n_cells})"
        )


def _make_hex_mesh(
    nx: int = 3,
    ny: int = 3,
    nz: int = 3,
) -> _FakeUnstructuredGrid:
    """작은 육면체(hexahedral) 메쉬를 NumPy로 생성한다.

    Args:
        nx: x 방향 격자 수.
        ny: y 방향 격자 수.
        nz: z 방향 격자 수.

    Returns:
        _FakeUnstructuredGrid 인스턴스.
    """
    # 노드 좌표 생성
    xs = np.linspace(0.0, 1.0, nx + 1)
    ys = np.linspace(0.0, 1.0, ny + 1)
    zs = np.linspace(0.0, 1.0, nz + 1)
    xg, yg, zg = np.meshgrid(xs, ys, zs, indexing="ij")
    points = np.column_stack([xg.ravel(), yg.ravel(), zg.ravel()])  # (n_pts, 3)

    def idx(i: int, j: int, k: int) -> int:
        return i * (ny + 1) * (nz + 1) + j * (nz + 1) + k

    cells_list: list[int] = []
    cell_types_list: list[int] = []
    VTK_HEXAHEDRON = 12  # VTK 육면체 셀 타입

    for i in range(nx):
        for j in range(ny):
            for k in range(nz):
                # 8개 꼭짓점 인덱스 (VTK hexahedron 순서)
                v = [
                    idx(i, j, k),
                    idx(i + 1, j, k),
                    idx(i + 1, j + 1, k),
                    idx(i, j + 1, k),
                    idx(i, j, k + 1),
                    idx(i + 1, j, k + 1),
                    idx(i + 1, j + 1, k + 1),
                    idx(i, j + 1, k + 1),
                ]
                cells_list.extend([8, *v])  # VTK 형식: 셀 크기 + 인덱스
                cell_types_list.append(VTK_HEXAHEDRON)

    mesh = _FakeUnstructuredGrid(
        points=points.astype(np.float64),
        cells=np.array(cells_list, dtype=np.int64),
        cell_types=np.array(cell_types_list, dtype=np.uint8),
    )

    # 샘플 필드 데이터 추가
    rng = np.random.default_rng(42)
    mesh.point_data["U"] = rng.standard_normal((mesh.n_points, 3))  # 속도장 (벡터)
    mesh.point_data["p"] = rng.standard_normal(mesh.n_points)  # 압력장 (스칼라)
    mesh.point_data["T"] = 300.0 + rng.standard_normal(mesh.n_points) * 10.0  # 온도장

    return mesh


# ---------------------------------------------------------------------------
# pytest 픽스처
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def sample_mesh() -> _FakeUnstructuredGrid:
    """작은 3×3×3 육면체 메쉬를 반환하는 세션 스코프 픽스처.

    pyvista 없이도 동작한다. 점 데이터에는 "U" (속도), "p" (압력),
    "T" (온도) 필드가 미리 채워져 있다.

    Returns:
        27개 점, 8개 셀을 가진 가짜 UnstructuredGrid.

    Examples:
        >>> def test_something(sample_mesh):
        ...     assert sample_mesh.n_points == 64  # 4x4x4 노드
        ...     assert "U" in sample_mesh.point_data
    """
    return _make_hex_mesh(nx=3, ny=3, nz=3)


@pytest.fixture(scope="session")
def sample_snapshots(sample_mesh: _FakeUnstructuredGrid) -> np.ndarray:
    """랜덤 스냅샷 행렬을 반환하는 세션 스코프 픽스처.

    ``sample_mesh`` 의 점 수를 기반으로 (n_samples, n_features) 형태의
    NumPy 배열을 생성한다.

    Args:
        sample_mesh: :func:`sample_mesh` 픽스처에서 제공되는 메쉬.

    Returns:
        shape = (20, n_points * 3) 의 float64 배열.
        (20개 타임스텝, 속도 3성분 × n_points 특성)
    """
    rng = np.random.default_rng(123)
    n_samples = 20
    n_features = sample_mesh.n_points * 3  # U_x, U_y, U_z
    return rng.standard_normal((n_samples, n_features)).astype(np.float64)


@pytest.fixture
def small_xy_dataset() -> tuple[np.ndarray, np.ndarray]:
    """소규모 입출력 데이터셋을 반환하는 픽스처.

    대리 모델(surrogate) 학습 테스트에 사용한다.

    Returns:
        (X, y) 튜플.
        X: shape = (50, 3) — 3차원 설계 변수.
        y: shape = (50, 2) — 2차원 응답값.
    """
    rng = np.random.default_rng(7)
    X = rng.uniform(-1.0, 1.0, (50, 3))
    # y = 비선형 함수 + 노이즈
    y = np.column_stack([
        np.sin(X[:, 0]) * np.cos(X[:, 1]) + 0.05 * rng.standard_normal(50),
        X[:, 2] ** 2 - X[:, 0] * X[:, 1] + 0.05 * rng.standard_normal(50),
    ])
    return X.astype(np.float64), y.astype(np.float64)
