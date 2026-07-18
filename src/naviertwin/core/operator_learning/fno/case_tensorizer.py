"""형상 가변 케이스 세트 → 공통 격자 텐서 변환 (GeometryFNO2D 입력 준비).

케이스마다 메쉬(형상)가 다른 정상 케이스 세트를 **하나의 공통 균일 격자** 위
(N, H, W, C) 텐서로 바꾼다. 형상은 부호거리(SDF)·유체 마스크 채널로, 운전조건
파라미터는 상수 브로드캐스트 채널로 들어간다 — DeepCFD(Ribeiro et al., 2020)·
Thuerey et al.(2020) 계열의 표준 입력 인코딩이다.

공통 격자 구성(합집합 바운딩 박스 위 등방 격자, ``grid.sample`` +
``vtkValidPointMask``, EDT 기반 SDF)은 :func:`naviertwin.web.service.
resample_cases_to_common_grid` 와 같은 레시피다 — core 는 web 에 의존할 수
없으므로(core→web 역참조 금지) 최소 로직을 여기에 복제했다.

Examples:
    >>> from naviertwin.core.operator_learning.fno.case_tensorizer import (
    ...     cases_to_grid_tensors,
    ... )
    >>> result = cases_to_grid_tensors(
    ...     datasets, params, field_names=["p"], resolution=48
    ... )                                          # doctest: +SKIP
    >>> result["inputs"].shape                     # doctest: +SKIP
    (5, 33, 49, 3)  # [sdf, mask, mu_0]
"""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np
from numpy.typing import NDArray

from naviertwin.utils.logger import get_logger

logger = get_logger(__name__)

_COMPONENT_SUFFIXES = ("x", "y", "z")


def _uniform_grid_over_bounds(bounds: Sequence[float], resolution: int) -> Any:
    """바운딩 박스를 덮는 등방 균일 격자를 만든다 (두께 0 축은 1칸 → 2D 자동).

    web/service 의 ``_uniform_grid_over`` 와 같은 규칙(2% 패딩, 최장 축 기준
    등방 간격)을 core 쪽에 복제한 것이다.

    Args:
        bounds: (xmin, xmax, ymin, ymax, zmin, zmax).
        resolution: 최장 축 방향 격자 분할 수.

    Returns:
        pyvista ImageData 격자.

    Raises:
        ValueError: 바운딩 박스가 비어 있는 경우.
    """
    import pyvista as pv

    box = np.asarray(bounds, dtype=np.float64)
    lo = box[[0, 2, 4]]
    hi = box[[1, 3, 5]]
    span = hi - lo
    pad = np.where(span > 0, span * 0.02, 0.0)
    lo, hi = lo - pad, hi + pad
    span = hi - lo
    longest = float(span.max())
    if longest <= 0:
        raise ValueError("바운딩 박스가 비어 있어 격자를 만들 수 없습니다.")
    step = longest / max(2, int(resolution))
    dims = [max(1, int(round(s / step)) + 1) if s > 0 else 1 for s in span]
    spacing = [step if s > 0 else 1.0 for s in span]
    return pv.ImageData(
        dimensions=tuple(dims), spacing=tuple(spacing), origin=tuple(lo)
    )


def _field_channels(
    sampled: Any, name: str
) -> list[tuple[str, NDArray[np.float64]]]:
    """샘플된 격자에서 필드 하나를 (채널명, 평탄 배열) 목록으로 푼다.

    벡터 필드(``U`` 등)는 성분별 채널(``U_x`` 등)로 확장한다.

    Raises:
        ValueError: 필드가 샘플 결과에 없는 경우.
    """
    if name not in sampled.point_data:
        available = list(sampled.point_data.keys())
        raise ValueError(
            f"필드 '{name}' 이(가) 샘플 결과에 없습니다. 사용 가능: {available}"
        )
    values = np.asarray(sampled.point_data[name], dtype=np.float64)
    if values.ndim == 1:
        return [(name, values)]
    channels: list[tuple[str, NDArray[np.float64]]] = []
    for j in range(values.shape[1]):
        suffix = (
            _COMPONENT_SUFFIXES[j] if values.shape[1] <= 3 else str(j)
        )
        channels.append((f"{name}_{suffix}", values[:, j]))
    return channels


def cases_to_grid_tensors(
    datasets: Sequence[Any],
    params: Any,
    field_names: Sequence[str],
    resolution: int = 48,
    *,
    param_names: Sequence[str] | None = None,
) -> dict[str, Any]:
    """형상 가변 케이스들을 공통 균일 격자 위 (N, H, W, C) 텐서로 바꾼다.

    입력 채널 순서는 ``[sdf, mask, μ_1, ..., μ_k]`` 로 고정된다:

    - ``sdf``: EDT 부호거리 (+ = 유체, − = 고체/도메인 밖). 형상 정보.
    - ``mask``: 유체 1 / 고체 0. ``vtkValidPointMask`` 그대로.
    - ``μ_j``: 케이스 파라미터 값을 격자 전체에 상수로 깐 채널 (원시값 —
      표준화는 :class:`~naviertwin.core.operator_learning.fno.geometry_fno.
      GeometryFNO2D` 내부에서 수행).

    타깃 채널은 요청한 필드를 격자에 샘플한 값이다. 고체 내부 격자점은 0 으로
    채워진다(``mask`` 채널이 그 영역을 식별한다). 벡터 필드는 성분별 채널로
    확장된다 (예: ``U`` → ``U_x, U_y, U_z``).

    현재 2D 전용이다 — 격자에 두께 0 축이 정확히 하나 있어야 한다.

    Args:
        datasets: 케이스 목록. ``mesh`` 속성을 가진 CFDDataset 또는 pyvista
            메쉬 자체. 케이스마다 메쉬(형상)가 달라도 된다.
        params: 케이스별 운전조건 파라미터. shape = (N, k) 또는 (N,).
            k = 0 (파라미터 없음, 형상만)도 허용.
        field_names: 타깃으로 샘플할 필드 이름 목록 (예: ``["p"]``).
        resolution: 최장 축 방향 격자 분할 수.
        param_names: 파라미터 채널 이름. None 이면 ``mu_0, mu_1, ...``.

    Returns:
        다음 키를 담은 dict:

        - ``"inputs"``: (N, H, W, 2 + k) float32 — [sdf, mask, μ...].
        - ``"targets"``: (N, H, W, C_out) float32 — 샘플된 필드 채널.
        - ``"grid"``: 공통 pyvista ImageData 격자.
        - ``"channel_names"``: 입력 채널 이름 목록.
        - ``"meta"``: ``target_names``, ``dims``, ``spacing``, ``origin``,
          ``flat_axis``, ``hw``, ``resolution``, ``n_cases``, ``grid_summary``.

    Raises:
        ValueError: 케이스가 없거나, 파라미터 개수가 케이스 수와 다르거나,
            격자가 2D(두께 0 축 정확히 하나)가 아니거나, 필드가 없는 경우.
    """
    from scipy.ndimage import distance_transform_edt

    meshes = [getattr(case, "mesh", case) for case in datasets]
    n_cases = len(meshes)
    if n_cases == 0:
        raise ValueError("텐서로 만들 케이스가 없습니다.")

    mu = np.asarray(params, dtype=np.float64)
    if mu.ndim == 1:
        mu = mu.reshape(-1, 1)
    if mu.size == 0:
        mu = mu.reshape(n_cases, 0)
    if mu.ndim != 2 or mu.shape[0] != n_cases:
        raise ValueError(
            f"params 는 (N, k) 형태여야 합니다. N={n_cases}, 현재: {mu.shape}"
        )
    n_params = mu.shape[1]

    if param_names is not None and len(param_names) != n_params:
        raise ValueError(
            f"param_names 길이({len(param_names)})가 파라미터 수({n_params})와 다릅니다."
        )
    mu_names = (
        [str(n) for n in param_names]
        if param_names is not None
        else [f"mu_{j}" for j in range(n_params)]
    )

    # 모든 케이스를 덮는 합집합 바운딩 박스 위에 등방 격자를 만든다.
    box = np.asarray([mesh.bounds for mesh in meshes], dtype=np.float64)
    union = [
        box[:, 0].min(), box[:, 1].max(),
        box[:, 2].min(), box[:, 3].max(),
        box[:, 4].min(), box[:, 5].max(),
    ]
    grid = _uniform_grid_over_bounds(union, resolution)
    dims = [int(d) for d in grid.dimensions]
    spacing = [float(s) for s in grid.spacing]

    flat_axes = [i for i, d in enumerate(dims) if d == 1]
    if len(flat_axes) != 1:
        raise ValueError(
            "현재 2D 케이스만 지원합니다 — 두께 0 축이 정확히 하나여야 합니다. "
            f"격자 dims={tuple(dims)} (두께 0 축 {len(flat_axes)}개)."
        )
    flat_axis = flat_axes[0]
    # VTK point 순서는 (z, y, x) — reshape 후 두께 0 축을 squeeze 하면 (H, W).
    reversed_dims = tuple(reversed(dims))
    squeeze_axis = 2 - flat_axis
    hw = tuple(d for i, d in enumerate(reversed_dims) if i != squeeze_axis)
    height, width = int(hw[0]), int(hw[1])

    def to_hw(flat: NDArray[np.float64]) -> NDArray[np.float64]:
        return flat.reshape(reversed_dims).squeeze(axis=squeeze_axis)

    # EDT 는 격자 인덱스 공간에서 계산 → 축별 물리 간격으로 환산.
    # 배열이 (z, y, x) 순서이므로 spacing 도 뒤집는다 (web/service 와 동일).
    edt_sampling = list(reversed(spacing))

    inputs = np.zeros(
        (n_cases, height, width, 2 + n_params), dtype=np.float32
    )
    target_names: list[str] | None = None
    targets_list: list[NDArray[np.float32]] = []

    for i, mesh in enumerate(meshes):
        sampled = grid.sample(mesh)
        mask = np.asarray(
            sampled.point_data.get("vtkValidPointMask", np.ones(grid.n_points)),
            dtype=np.float64,
        )
        shaped = mask.reshape(reversed_dims)
        # 부호거리: 유체 안쪽 +(고체까지 거리), 고체 안쪽 −(유체까지 거리).
        d_fluid = distance_transform_edt(shaped, sampling=edt_sampling)
        d_solid = distance_transform_edt(1.0 - shaped, sampling=edt_sampling)
        sdf = (np.asarray(d_fluid) - np.asarray(d_solid)).reshape(-1)

        inputs[i, :, :, 0] = to_hw(sdf).astype(np.float32)
        inputs[i, :, :, 1] = to_hw(mask).astype(np.float32)
        for j in range(n_params):
            inputs[i, :, :, 2 + j] = np.float32(mu[i, j])

        case_channels: list[NDArray[np.float32]] = []
        case_names: list[str] = []
        for name in field_names:
            for channel_name, flat_values in _field_channels(sampled, name):
                # 고체 내부는 0 — sample 기본값이지만 마스크로 확정한다.
                zeroed = flat_values * mask
                case_channels.append(to_hw(zeroed).astype(np.float32))
                case_names.append(channel_name)
        if target_names is None:
            target_names = case_names
        elif case_names != target_names:
            raise ValueError(
                f"케이스 {i} 의 타깃 채널({case_names})이 첫 케이스"
                f"({target_names})와 다릅니다."
            )
        targets_list.append(np.stack(case_channels, axis=-1))

    targets = np.stack(targets_list, axis=0).astype(np.float32)
    channel_names = ["sdf", "mask", *mu_names]
    summary = (
        f"공통 격자 {'×'.join(str(d) for d in dims)} ({grid.n_points:,}점)"
        f" · {n_cases}케이스 · 입력 채널 {channel_names}"
    )
    logger.info("케이스 텐서화 완료: %s", summary)
    return {
        "inputs": inputs,
        "targets": targets,
        "grid": grid,
        "channel_names": channel_names,
        "meta": {
            "target_names": list(target_names or []),
            "dims": tuple(dims),
            "spacing": tuple(spacing),
            "origin": tuple(float(o) for o in grid.origin),
            "flat_axis": int(flat_axis),
            "hw": (height, width),
            "resolution": int(resolution),
            "n_cases": int(n_cases),
            "grid_summary": summary,
        },
    }


__all__ = ["cases_to_grid_tensors"]
