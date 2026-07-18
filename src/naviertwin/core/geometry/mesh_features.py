"""메쉬 기반 벽면 특징(wall feature) 전처리 모듈.

임의의 볼륨 메쉬와 벽면 표면(PolyData)으로부터 벽 거리(wall distance),
부호 있는 거리(SDF), 표면 기하 특징(법선·곡률·면적)을 계산한다.
:mod:`naviertwin.core.geometry.sdf` 의 해석적 프리미티브와 달리 임의의
표면 메쉬에 대해 동작하므로, OpenFOAM 리더가 보존한 경계 패치
(``metadata["boundary_patch_meshes"]``)를 그대로 벽면으로 사용할 수 있다.

Examples:
    >>> import pyvista as pv
    >>> from naviertwin.core.geometry.mesh_features import wall_distance
    >>> grid = pv.ImageData(dimensions=(8, 8, 8))
    >>> wall = pv.Sphere(radius=2.0, center=(3.5, 3.5, 3.5))
    >>> d = wall_distance(grid, wall)
    >>> d.shape == (grid.n_points,)
    True
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

import numpy as np
from numpy.typing import NDArray

from naviertwin.utils.logger import get_logger

if TYPE_CHECKING:
    import pyvista as pv

    from naviertwin.core.cfd_reader.base import CFDDataset

logger = get_logger(__name__)

__all__ = [
    "attach_wall_features",
    "signed_distance",
    "surface_features",
    "wall_distance",
    "wall_surface_from_patches",
]


# ---------------------------------------------------------------------------
# 내부 헬퍼
# ---------------------------------------------------------------------------


def _as_polydata_surface(surface: Any) -> pv.PolyData:
    """임의의 pyvista DataSet 을 PolyData 표면으로 변환한다.

    Args:
        surface: pyvista DataSet (PolyData, UnstructuredGrid 등).

    Returns:
        PolyData 표면.

    Raises:
        ImportError: pyvista 가 설치되어 있지 않은 경우.
    """
    try:
        import pyvista as pv
    except ImportError as exc:
        raise ImportError("pyvista 가 필요합니다: pip install pyvista") from exc

    if isinstance(surface, pv.PolyData):
        return surface
    return surface.extract_surface()


def _implicit_distance(mesh: Any, wall_surface: Any) -> NDArray[np.float64]:
    """메쉬 각 점에서 벽면까지의 부호 있는 implicit distance 를 계산한다.

    Args:
        mesh: 임의의 pyvista DataSet (볼륨 메쉬).
        wall_surface: 벽면 표면 (PolyData 로 변환 가능해야 한다).

    Returns:
        shape = (n_points,) 의 부호 있는 거리 배열.
    """
    surface = _as_polydata_surface(wall_surface)
    result = mesh.compute_implicit_distance(surface)
    return np.asarray(result.point_data["implicit_distance"], dtype=np.float64)


def _is_closed_surface(surface: Any) -> bool:
    """표면이 닫혀 있는지(경계 에지가 없는지) 판별한다.

    Args:
        surface: pyvista PolyData 표면.

    Returns:
        경계(open) 에지가 하나도 없으면 True.
    """
    try:
        return int(surface.n_open_edges) == 0
    except Exception:  # noqa: BLE001 — n_open_edges 미지원 타입 폴백
        pass
    try:
        edges = surface.extract_feature_edges(
            boundary_edges=True,
            feature_edges=False,
            manifold_edges=False,
            non_manifold_edges=False,
        )
        return int(edges.n_cells) == 0
    except Exception:  # noqa: BLE001 — 판별 불가 시 보수적으로 열림 처리
        return False


# ---------------------------------------------------------------------------
# 공개 API
# ---------------------------------------------------------------------------


def wall_distance(mesh: Any, wall_surface: Any) -> NDArray[np.float64]:
    """메쉬 각 점에서 벽면까지의 거리(항상 비음수)를 계산한다.

    pyvista ``compute_implicit_distance`` 기반이므로 정렬/비정렬을 가리지
    않는 임의의 볼륨 메쉬와 임의의 PolyData 벽면 조합에 대해 동작한다.

    Args:
        mesh: 임의의 pyvista DataSet (볼륨 메쉬).
        wall_surface: 벽면 표면 (PolyData 또는 표면 추출 가능한 DataSet).

    Returns:
        shape = (mesh.n_points,) 의 비음수 거리 배열.
    """
    return np.abs(_implicit_distance(mesh, wall_surface))


def signed_distance(mesh: Any, wall_surface: Any) -> NDArray[np.float64]:
    """메쉬 각 점에서 벽면까지의 부호 있는 거리(SDF)를 계산한다.

    부호는 벽면 법선(pseudonormal) 방향으로 결정된다. 바깥쪽 법선을 가진
    **닫힌 표면**에서는 내부가 음수, 외부가 양수이다.

    Warning:
        열린(open) 표면에서는 부호를 신뢰할 수 없다. 경계 에지 근처에서
        법선 기준 부호가 임의로 뒤집힐 수 있으므로, 열린 표면에서는
        크기(절대값)만 유효하다. 부호까지 필요하면 닫힌 표면을 사용하거나
        :func:`wall_distance` 를 사용할 것.

    Args:
        mesh: 임의의 pyvista DataSet (볼륨 메쉬).
        wall_surface: 벽면 표면 (PolyData 또는 표면 추출 가능한 DataSet).

    Returns:
        shape = (mesh.n_points,) 의 부호 있는 거리 배열.
    """
    return _implicit_distance(mesh, wall_surface)


def surface_features(wall_surface: Any) -> dict[str, NDArray[np.float64]]:
    """벽면 표면의 기하 특징(법선, 곡률, 면적)을 계산한다.

    Args:
        wall_surface: 벽면 표면 (PolyData 또는 표면 추출 가능한 DataSet).

    Returns:
        딕셔너리:
            - ``"normals"``: shape = (n_points, 3) 점 법선.
            - ``"curvature"``: shape = (n_points,) 평균 곡률.
            - ``"area"``: shape = (n_cells,) 셀(면) 면적.
    """
    surface = _as_polydata_surface(wall_surface)

    with_normals = surface.compute_normals(
        cell_normals=False, point_normals=True
    )
    normals = np.asarray(with_normals.point_data["Normals"], dtype=np.float64)

    curvature = np.asarray(surface.curvature("mean"), dtype=np.float64)

    sized = surface.compute_cell_sizes(length=False, area=True, volume=False)
    area = np.asarray(sized.cell_data["Area"], dtype=np.float64)

    return {"normals": normals, "curvature": curvature, "area": area}


def attach_wall_features(
    dataset: CFDDataset, wall_surface: Any, *, prefix: str = "wall"
) -> list[str]:
    """데이터셋 메쉬에 벽 거리 필드를 point_data 로 부착한다.

    항상 ``f"{prefix}_distance"`` (비음수 거리)를 부착하고, 벽면이 닫힌
    표면일 때만 ``f"{prefix}_sdf"`` (부호 있는 거리)를 추가로 부착한다.
    열린 표면에서는 부호를 신뢰할 수 없으므로 SDF 를 생략한다
    (:func:`signed_distance` 의 Warning 참조).

    부착된 필드 이름은 ``dataset.field_names`` 에도 추가되어 이후 학습
    파이프라인에서 일반 필드처럼 입력으로 선택할 수 있다.

    Args:
        dataset: 대상 :class:`~naviertwin.core.cfd_reader.base.CFDDataset`.
        wall_surface: 벽면 표면 (PolyData 또는 표면 추출 가능한 DataSet).
        prefix: 부착 필드 이름 접두사. 기본값 ``"wall"``.

    Returns:
        부착된 필드 이름 리스트 (예: ``["wall_distance", "wall_sdf"]``).
    """
    surface = _as_polydata_surface(wall_surface)
    signed = _implicit_distance(dataset.mesh, surface)

    attached: list[str] = []

    distance_name = f"{prefix}_distance"
    dataset.mesh.point_data[distance_name] = np.abs(signed)
    attached.append(distance_name)

    if _is_closed_surface(surface):
        sdf_name = f"{prefix}_sdf"
        dataset.mesh.point_data[sdf_name] = signed
        attached.append(sdf_name)
    else:
        logger.warning(
            "벽면 표면이 열려 있어 부호 있는 거리('%s_sdf')는 생략합니다.",
            prefix,
        )

    for name in attached:
        if name not in dataset.field_names:
            dataset.field_names.append(name)

    logger.info("벽면 특징 부착 완료: %s", attached)
    return attached


def wall_surface_from_patches(
    dataset: CFDDataset, patch_names: Sequence[str]
) -> pv.PolyData:
    """경계 패치 메타데이터에서 선택한 패치들을 하나의 벽면으로 병합한다.

    OpenFOAM 리더가 ``metadata["boundary_patch_meshes"]`` 에 보존한
    패치 표면들 중 ``patch_names`` 로 지정한 것들을 병합해 단일 PolyData
    벽면을 만든다. 결과는 :func:`attach_wall_features` 등의 입력으로
    바로 사용할 수 있다.

    Args:
        dataset: 경계 패치 메타데이터를 가진 데이터셋.
        patch_names: 병합할 패치 이름 목록.

    Returns:
        병합된 PolyData 벽면.

    Raises:
        ValueError: 경계 패치 메타데이터가 없거나, patch_names 가 비었거나,
            알 수 없는 패치 이름이 포함된 경우.
    """
    try:
        import pyvista as pv
    except ImportError as exc:
        raise ImportError("pyvista 가 필요합니다: pip install pyvista") from exc

    patch_meshes = dataset.metadata.get("boundary_patch_meshes")
    if not isinstance(patch_meshes, dict) or not patch_meshes:
        raise ValueError(
            "데이터셋 metadata 에 'boundary_patch_meshes' 가 없습니다. "
            "경계 패치를 보존하는 리더(예: OpenFOAM pyvista 리더)로 "
            "읽었는지 확인하세요."
        )

    names = list(patch_names)
    if not names:
        raise ValueError(
            "patch_names 가 비어 있습니다. 병합할 패치 이름을 1개 이상 "
            "지정하세요."
        )

    unknown = [name for name in names if name not in patch_meshes]
    if unknown:
        raise ValueError(
            f"알 수 없는 패치 이름: {unknown}. "
            f"사용 가능한 패치: {sorted(patch_meshes)}"
        )

    surfaces = [_as_polydata_surface(patch_meshes[name]) for name in names]
    if len(surfaces) == 1:
        merged: Any = surfaces[0].copy()
    else:
        merged = pv.merge(surfaces)

    if not isinstance(merged, pv.PolyData):
        merged = merged.extract_surface()
    return merged
