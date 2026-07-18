"""Qt 비의존 CFD 렌더 메쉬 준비 유틸 (웹/trame 뷰어용).

데스크톱 :class:`naviertwin.gui.widgets.vtk_viewer.VtkViewer` 에서 검증된
mesh/field 준비 로직을 Qt 의존성 없이 순수 함수로 추출한 모듈이다. 웹(trame)
뷰어가 동일한 PyVista 렌더 파이프라인을 재사용하도록 한다.

핵심 함수:
    - :func:`prepare_render_mesh`: ``CFDDataset`` + field + timestep →
      ``(render_mesh, scalar_name)`` 튜플. trame ``plotter_ui`` 가 그대로 렌더.
    - :func:`field_names_from_mesh`: 메쉬의 point/cell field 이름 목록.

설계 원칙:
    - UnstructuredGrid 는 렌더 직전 surface 추출 (VTK dangling reference 회피).
    - 벡터 필드는 ``<field>__mag`` magnitude scalar 로 컬러링.
    - Qt/GUI 위젯 상태가 아닌 명시적 인자로만 동작 (순수 함수).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Optional

import numpy as np

if TYPE_CHECKING:
    from naviertwin.core.cfd_reader.base import CFDDataset

# 데스크톱 뷰어와 동일한 컬러맵/색상 팔레트.
COLORMAPS = ["coolwarm", "viridis", "plasma", "turbo", "jet", "rainbow", "gray"]
SOLID_COLOR = "#c9d1d9"
EDGE_COLOR = "#1a1f24"


def _pyvista() -> Any:
    """PyVista 모듈을 지연 import 한다 (없으면 None)."""
    try:
        import pyvista as pv
    except Exception:  # noqa: BLE001 — pyvista/VTK 초기화 환경 의존
        return None
    return pv


def field_names_from_mesh(mesh: Any) -> list[str]:
    """메쉬의 point_data / cell_data field 이름 목록을 반환한다."""
    names: list[str] = []
    try:
        point_keys = tuple(mesh.point_data.keys())
        point_index = 0
        while point_index < len(point_keys):
            names.append(str(point_keys[point_index]))
            point_index += 1
        cell_keys = tuple(mesh.cell_data.keys())
        cell_index = 0
        while cell_index < len(cell_keys):
            name = str(cell_keys[cell_index])
            if name not in names:
                names.append(name)
            cell_index += 1
    except Exception:  # noqa: BLE001
        pass
    return names


def available_fields(dataset: CFDDataset) -> list[str]:
    """dataset.field_names + 메쉬 직접 field + 시계열 field 를 합친 목록."""
    names = list(getattr(dataset, "field_names", []) or [])
    mesh_names = field_names_from_mesh(getattr(dataset, "mesh", None))
    name_index = 0
    while name_index < len(mesh_names):
        if mesh_names[name_index] not in names:
            names.append(mesh_names[name_index])
        name_index += 1
    time_series = _time_series_fields(dataset)
    for key in time_series:
        if key not in names:
            names.append(key)
    return names


def preferred_field(names: list[str]) -> str:
    """표시 우선순위(p, pressure, T, U, velocity)에 따라 기본 field 를 고른다."""
    if not names:
        return ""
    preferred_names = ("p", "pressure", "T", "U", "velocity")
    cursor = 0
    while cursor < len(preferred_names):
        if preferred_names[cursor] in names:
            return preferred_names[cursor]
        cursor += 1
    return names[0]


def _time_series_fields(dataset: CFDDataset) -> dict[str, Any]:
    meta = getattr(dataset, "metadata", {}) or {}
    series = meta.get("time_series_fields", {})
    return series if isinstance(series, dict) else {}


def _select_timestep_array(dataset: CFDDataset, arr: np.ndarray, timestep: int) -> np.ndarray:
    """시계열 배열에서 현재 timestep 슬라이스를 추출한다."""
    n_steps = max(1, int(getattr(dataset, "n_time_steps", 1)))
    step = min(max(0, timestep), max(0, n_steps - 1))
    n_points = int(getattr(dataset, "n_points", 0))
    n_cells = int(getattr(dataset, "n_cells", 0))

    if n_steps <= 1:
        return arr
    if arr.ndim >= 2 and arr.shape[0] == n_steps:
        return arr[step]
    if arr.ndim >= 2 and arr.shape[1] == n_steps and arr.shape[0] in {n_points, n_cells}:
        return arr[:, step]
    if arr.ndim == 1 and arr.size % n_steps == 0:
        per_step = arr.size // n_steps
        return arr[step * per_step : (step + 1) * per_step]
    if arr.ndim == 2 and arr.shape[0] in {n_points * n_steps, n_cells * n_steps}:
        per_step = arr.shape[0] // n_steps
        return arr[step * per_step : (step + 1) * per_step]
    return arr


def _normalize_field_array(arr: np.ndarray, mesh: Any) -> tuple[Optional[np.ndarray], Optional[str]]:
    """배열을 point/cell 길이에 맞춰 정규화하고 위치를 판정한다."""
    n_points = int(getattr(mesh, "n_points", 0))
    n_cells = int(getattr(mesh, "n_cells", 0))
    values = np.asarray(arr)

    if values.ndim >= 3 and values.shape[-1] in {2, 3}:
        values = values.reshape(-1, values.shape[-1])
    elif values.ndim >= 2 and values.shape[0] not in {n_points, n_cells}:
        if values.size == n_points:
            values = values.reshape(n_points)
        elif values.size == n_cells:
            values = values.reshape(n_cells)
        elif values.shape[-1] in {2, 3} and values.size % values.shape[-1] == 0:
            values = values.reshape(-1, values.shape[-1])
        else:
            values = values.reshape(-1)

    if values.shape[0] == n_points:
        return values, "point"
    if values.shape[0] == n_cells:
        return values, "cell"
    return None, None


def _field_array_for_step(
    dataset: CFDDataset,
    mesh: Any,
    field_name: str,
    timestep: int,
) -> tuple[Optional[np.ndarray], Optional[str]]:
    time_series = _time_series_fields(dataset)
    if field_name in time_series:
        arr = np.asarray(time_series[field_name])
        arr = _select_timestep_array(dataset, arr, timestep)
        return _normalize_field_array(arr, mesh)
    if field_name in getattr(mesh, "point_data", {}):
        return np.asarray(mesh.point_data[field_name]), "point"
    if field_name in getattr(mesh, "cell_data", {}):
        return np.asarray(mesh.cell_data[field_name]), "cell"
    return None, None


def _attach_render_array(
    mesh: Any,
    field_name: str,
    arr: np.ndarray,
    location: str,
) -> str:
    """스칼라/벡터 배열을 메쉬에 붙이고 컬러링용 scalar 이름을 반환한다."""
    values = np.asarray(arr)
    scalar_name = field_name
    if values.ndim > 1 and values.shape[-1] > 1:
        scalar_name = f"{field_name}__mag"
        values = np.linalg.norm(values, axis=-1)
    if location == "point":
        mesh.point_data[scalar_name] = np.asarray(values, dtype=np.float64)
    else:
        mesh.cell_data[scalar_name] = np.asarray(values, dtype=np.float64)
    return scalar_name


def _surface_for_render(mesh: Any) -> Any:
    pv = _pyvista()
    if pv is None:
        return mesh
    try:
        if isinstance(mesh, pv.UnstructuredGrid):
            return mesh.extract_surface(algorithm="dataset_surface")
    except Exception:  # noqa: BLE001
        try:
            if isinstance(mesh, pv.UnstructuredGrid):
                return mesh.extract_surface()
        except Exception:  # noqa: BLE001
            return mesh
    return mesh


def _copy_mesh(mesh: Any) -> Any:
    try:
        return mesh.copy(deep=True)
    except Exception:  # noqa: BLE001
        return mesh


def prepare_render_mesh(
    dataset: CFDDataset,
    field_name: str = "",
    timestep: int = 0,
) -> tuple[Any, str]:
    """``CFDDataset`` 에서 렌더 가능한 (mesh, scalar_name) 을 준비한다.

    Args:
        dataset: 표시할 CFD 데이터셋.
        field_name: 컬러링할 field 이름 (빈 문자열이면 solid color).
        timestep: 시계열 데이터의 표시 timestep 인덱스.

    Returns:
        ``(render_mesh, scalar_name)``. ``scalar_name`` 이 빈 문자열이면
        solid color 로 렌더해야 한다.
    """
    if dataset is None:
        raise ValueError("dataset is not loaded")

    mesh = _copy_mesh(dataset.mesh)
    scalar_name = ""
    if field_name:
        arr, location = _field_array_for_step(dataset, mesh, field_name, timestep)
        if arr is not None and location is not None:
            scalar_name = _attach_render_array(mesh, field_name, arr, location)

    render_mesh = _surface_for_render(mesh)
    return render_mesh, scalar_name


def mesh_is_flat(mesh: Any) -> bool:
    """메쉬가 평면(2D, z 두께 0)인지 판정한다 (카메라 기본 뷰 선택용)."""
    try:
        bounds = mesh.bounds
        z_span = float(bounds[5]) - float(bounds[4])
        return abs(z_span) < 1e-9
    except Exception:  # noqa: BLE001
        return False


__all__ = [
    "COLORMAPS",
    "EDGE_COLOR",
    "SOLID_COLOR",
    "available_fields",
    "field_names_from_mesh",
    "mesh_is_flat",
    "prepare_render_mesh",
    "preferred_field",
]
