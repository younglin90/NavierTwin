"""필드 의미론 — 보존량(conserved quantity) 필드 판별 + 근사 conservative resample.

해상도 낮추기·공통 격자 재샘플은 기본적으로 VTK ``sample()`` 점 보간을 쓴다.
온도·압력 같은 intensive 량은 점 보간이 타당하지만, 밀도·질량 유량·운동량 같은
보존량은 셀 부피 가중 없이 점 값만 보간하므로 재샘플 후 적분 총량이 보존되지
않는다. 진짜 conservative remapping(격자 겹침 부피로 재분배)은 ESMPy나
MEDCoupling 같은 무거운 GPL/LGPL 의존성이 필요해 이번 범위 밖이다 — 대신 이
모듈은 **가벼운 자체 근사**를 제공한다: 원본 점/셀을 target 격자점(래티스)에
부피(또는 점 개수) 가중 평균으로 모아 담는 방식이다. 진짜 supermesh 교차
기반 remap 은 아니지만, 점 보간보다 총량 보존이 훨씬 낫다(테스트로 증명 —
:mod:`tests.test_conservative_resample` 참고).

- :func:`flag_conserved_fields` — 보존량 의심 필드명 판별(경고용).
- :func:`conservative_resample_to_grid` — 근사 conservative resample 본체.
- :func:`total_field_integral` — 재샘플 전후 총량 비교용 헬퍼.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np

from naviertwin.utils.logger import get_logger

logger = get_logger(__name__)

# 보존량으로 의심되는 필드명 패턴 (소문자, 부분 일치).
#
# 보수적으로 명확한 것만 넣는다 — 과잉 경고보다 과소 경고가 낫다.
# 특히 "Q" 단독은 Q-criterion 과 혼동되므로 제외하고, 유량은
# "flux"/"flow_rate" 처럼 명시적인 이름만 잡는다. "energy" 단독도
# 비열·비내부에너지(intensive) 필드와 혼동될 수 있어 "total_energy" 만 잡는다.
CONSERVED_FIELD_PATTERNS: tuple[str, ...] = (
    "rho",  # 밀도(rho, rhoU, rhoE ...) — 부피 적분하면 질량/운동량/에너지
    "density",
    "mass",  # mass_flux, mass_flow, massFlowRate ...
    "flux",  # mass_flux, volumetric_flux, heat_flux(면적분량) ...
    "flow_rate",
    "flowrate",
    "momentum",
    "total_energy",
    "totalenergy",
    "enthalpy_flow",
)


def flag_conserved_fields(field_names: Sequence[str]) -> list[str]:
    """보존량으로 의심되는 필드명을 골라 반환한다.

    점 보간 재샘플은 셀 부피 가중 없이 점 값만 새 격자로 옮기므로, 부피(면적)
    적분으로 의미가 정의되는 보존량은 재샘플 후 총량(총질량·총유량 등)이
    보존되지 않는다. 총량 보존이 필요하면 격자 겹침 부피로 가중 재분배하는
    conservative remapping(ESMPy/MEDCoupling 계열)을 써야 한다.

    Args:
        field_names: 검사할 필드명 목록.

    Returns:
        ``CONSERVED_FIELD_PATTERNS`` 와 대소문자 무시 부분 일치하는 필드명
        목록 (입력 순서 유지).
    """
    flagged: list[str] = []
    for name in field_names:
        lowered = str(name).lower()
        if any(pattern in lowered for pattern in CONSERVED_FIELD_PATTERNS):
            flagged.append(str(name))
    return flagged


def _cell_weights(mesh: Any) -> np.ndarray | None:
    """mesh 의 셀 부피(3D) 또는 면적(2D) 배열을 반환한다. 실패하면 ``None``."""
    try:
        sized = mesh.compute_cell_sizes(length=False, area=True, volume=True)
    except Exception as exc:  # noqa: BLE001 — 셀 없는 point cloud 등 다양한 실패 허용
        logger.debug("compute_cell_sizes 실패: %s", exc)
        return None
    vol = np.asarray(sized.cell_data.get("Volume", []), dtype=np.float64)
    area = np.asarray(sized.cell_data.get("Area", []), dtype=np.float64)
    if vol.size and float(np.sum(vol)) > 0:
        return vol
    if area.size and float(np.sum(area)) > 0:
        return area
    return None


def conservative_resample_to_grid(
    mesh: Any, field_name: str, target_grid: Any
) -> np.ndarray:
    """부피(또는 점 개수) 가중 평균으로 ``field_name`` 을 ``target_grid`` 위로 재샘플한다.

    **이것은 근사다 — 진짜 conservative remap 이 아니다.** 진짜 supermesh
    교차 기반 conservative remapping(ESMPy/MEDCoupling 계열)은 target 셀과
    원본 셀의 실제 기하 겹침 부피를 계산해 가중한다. 여기서는 그 대신
    실용적인 근사를 쓴다: ``target_grid`` 는 균일 격자(``pv.ImageData``,
    ``origin``/``spacing``/``dimensions`` 를 가짐)라 가정하고, 원본 mesh 의
    각 점(또는 셀 중심)을 가장 가까운 target 격자점의 "구역"(한 변이
    ``spacing`` 인 격자 셀, 그 점이 중심)에 배정한 뒤, 그 구역에 배정된
    원본 값들을 셀 부피(가능하면) 또는 균등 가중(점 개수)으로 평균한다.

    점 보간(``grid.sample()``)은 target 점에서 원본을 하나만 보간해 셀
    내부의 다른 원본 점 정보를 버리므로 총량이 보존되지 않는다. 이 함수는
    구역 안의 모든 원본 값을 부피 가중으로 합쳐 담으므로 총 적분(질량 등)
    보존 오차가 점 보간보다 훨씬 작다 — 다만 여전히 근사이며 참값은 아니다
    (:func:`total_field_integral` 로 오차를 직접 비교할 수 있다).

    원본 mesh 에 셀 데이터가 있으면(``field_name`` 이 ``cell_data`` 에 있음)
    셀 부피를 자연 가중치로 쓴다. 점 데이터뿐이면 인접 셀 부피를 점으로
    평균해 가중치로 쓰고, 그마저 없으면(예: 셀 없는 point cloud) 균등
    가중(단순 점 개수 평균)으로 폴백한다.

    target 격자점 중 배정된 원본 점이 하나도 없는 구역(예: 원본보다 훨씬
    성긴 격자이거나 도메인 경계)은 최근접 원본 점 값으로 폴백하고 로그를
    남긴다.

    Args:
        mesh: 원본 ``pyvista.DataSet`` (point_data 또는 cell_data 에
            ``field_name`` 을 가짐).
        field_name: 재샘플할 필드명.
        target_grid: 균일 배경 격자(``pv.ImageData``) — ``origin``,
            ``spacing``, ``dimensions``, ``n_points``, ``points`` 필요.

    Returns:
        ``target_grid.n_points`` 길이의 배열 (스칼라) 또는
        ``(n_points, n_components)`` 배열 (벡터장).

    Raises:
        KeyError: ``field_name`` 이 mesh 의 point_data/cell_data 어디에도
            없는 경우.
        ValueError: 원본 mesh 에 점/셀이 하나도 없는 경우.
    """
    has_cell = field_name in getattr(mesh, "cell_data", {})
    has_point = field_name in getattr(mesh, "point_data", {})
    if not has_cell and not has_point:
        raise KeyError(
            f"필드 '{field_name}' 이 mesh 의 point_data/cell_data 어디에도 없습니다."
        )

    if has_cell:
        values = np.asarray(mesh.cell_data[field_name], dtype=np.float64)
        weights = _cell_weights(mesh)
        try:
            locations = np.asarray(mesh.cell_centers().points, dtype=np.float64)
        except Exception as exc:  # noqa: BLE001
            logger.debug("cell_centers() 실패, point 좌표로 폴백: %s", exc)
            locations = np.asarray(mesh.points, dtype=np.float64)[: len(values)]
    else:
        values = np.asarray(mesh.point_data[field_name], dtype=np.float64)
        locations = np.asarray(mesh.points, dtype=np.float64)
        weights = None
        cell_w = _cell_weights(mesh)
        if cell_w is not None:
            try:
                sized = mesh.compute_cell_sizes(length=False, area=True, volume=True)
                sized.cell_data["_ntwin_cw"] = cell_w
                as_point = sized.cell_data_to_point_data()
                point_w = np.asarray(as_point.point_data["_ntwin_cw"], dtype=np.float64)
                if np.any(point_w > 0):
                    weights = point_w
            except Exception as exc:  # noqa: BLE001
                logger.debug("cell_data_to_point_data 가중치 변환 실패: %s", exc)

    n_locations = int(locations.shape[0])
    if n_locations == 0:
        raise ValueError("원본 mesh 에 점(또는 셀)이 없어 재샘플할 수 없습니다.")
    if values.shape[0] != n_locations:
        # 셀 데이터인데 cell_centers 가 실패해 point 좌표로 폴백한 경우 등 —
        # 짧은 쪽 길이로 맞춘다 (그래도 안전하게 동작하도록).
        n_locations = min(n_locations, values.shape[0])
        locations = locations[:n_locations]
        values = values[:n_locations]

    if weights is None:
        weights = np.ones(n_locations, dtype=np.float64)
    else:
        weights = np.asarray(weights, dtype=np.float64).reshape(-1)
        if weights.shape[0] != n_locations:
            weights = np.ones(n_locations, dtype=np.float64)
    weights = np.where(np.isfinite(weights) & (weights > 0), weights, 0.0)
    if not np.any(weights > 0):
        weights = np.ones(n_locations, dtype=np.float64)

    dims = np.asarray(target_grid.dimensions, dtype=np.int64)
    origin = np.asarray(target_grid.origin, dtype=np.float64)
    spacing = np.asarray(target_grid.spacing, dtype=np.float64)
    spacing_safe = np.where(spacing > 0, spacing, 1.0)

    rel = (locations[:, :3] - origin) / spacing_safe
    idx = np.rint(rel).astype(np.int64)
    idx = np.clip(idx, 0, np.maximum(dims - 1, 0))
    linear = idx[:, 0] + idx[:, 1] * dims[0] + idx[:, 2] * dims[0] * dims[1]

    n_target = int(target_grid.n_points)
    ncomp = 1 if values.ndim == 1 else int(values.shape[1])
    values2d = values.reshape(n_locations, ncomp)

    sum_w = np.zeros(n_target, dtype=np.float64)
    np.add.at(sum_w, linear, weights)
    sum_wf = np.zeros((n_target, ncomp), dtype=np.float64)
    for c in range(ncomp):
        np.add.at(sum_wf[:, c], linear, weights * values2d[:, c])

    has_data = sum_w > 0
    result = np.full((n_target, ncomp), np.nan, dtype=np.float64)
    result[has_data] = sum_wf[has_data] / sum_w[has_data][:, None]

    n_empty = int((~has_data).sum())
    if n_empty > 0:
        logger.warning(
            "conservative_resample_to_grid: target 격자점 %d/%d 개에 배정된 원본 "
            "점이 없어 최근접 점 값으로 폴백합니다 (field='%s').",
            n_empty,
            n_target,
            field_name,
        )
        from scipy.spatial import cKDTree

        tree = cKDTree(locations[:, :3])
        target_points = np.asarray(target_grid.points, dtype=np.float64)[~has_data]
        _, nn_idx = tree.query(target_points)
        result[~has_data] = values2d[nn_idx]

    return result[:, 0] if ncomp == 1 else result


def total_field_integral(mesh: Any, field_name: str) -> float:
    """필드의 총 적분(부피 가중 합)을 근사한다 — 재샘플 전후 총량 비교용.

    셀 부피(3D, ``compute_cell_sizes`` 의 ``Volume``) 또는 면적(2D,
    ``Area``)으로 가중해 합산한다. 점 데이터는 셀로 평균 변환
    (``point_data_to_cell_data``) 후 합산한다. 벡터장은 성분별 노름
    (``np.linalg.norm``)으로 축약한다.

    ``mesh_processor.quality_report`` 의 부피 계산과 같은
    ``compute_cell_sizes`` 패턴을 재사용한다.

    Args:
        mesh: 대상 ``pyvista.DataSet``.
        field_name: 적분할 필드명 (point_data 또는 cell_data).

    Returns:
        총 적분값 (float). 셀 부피/면적을 하나도 못 구하면 0.0.

    Raises:
        KeyError: ``field_name`` 이 mesh 에 없는 경우.
    """
    has_cell = field_name in getattr(mesh, "cell_data", {})
    has_point = field_name in getattr(mesh, "point_data", {})
    if not has_cell and not has_point:
        raise KeyError(
            f"필드 '{field_name}' 이 mesh 의 point_data/cell_data 어디에도 없습니다."
        )

    sized = mesh.compute_cell_sizes(length=False, area=True, volume=True)
    vol = np.asarray(sized.cell_data.get("Volume", []), dtype=np.float64)
    area = np.asarray(sized.cell_data.get("Area", []), dtype=np.float64)
    weight = vol if float(np.sum(vol)) > 0 else area
    if weight.size == 0:
        return 0.0

    if has_cell:
        values = np.asarray(mesh.cell_data[field_name], dtype=np.float64)
    else:
        converted = sized.point_data_to_cell_data()
        values = np.asarray(converted.cell_data[field_name], dtype=np.float64)

    if values.ndim > 1:
        values = np.linalg.norm(values, axis=1)

    n = min(values.shape[0], weight.shape[0])
    return float(np.sum(values[:n] * weight[:n]))
