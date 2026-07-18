"""케이스(메쉬) → 그래프 변환 — 메쉬 네이티브(Route 2) 학습의 공통 배관.

케이스마다 메쉬(형상)가 달라도, 각 케이스를 **자기 격자 그대로** 그래프
하나로 바꾼다 — 공통 격자 재샘플이 없다는 것이 :mod:`~naviertwin.core.
operator_learning.fno.case_tensorizer` (Route 1, 균일 격자 텐서화)와 갈리는
지점이다. 진짜 구멍(장애물 자리에 셀 없음)이 그대로 보존된다.

규약(케이스 텐서화와의 일치):
    - 벡터 필드 성분 전개는 ``case_tensorizer._field_channels`` 와 같은 규칙
      (``U`` → ``U_x, U_y, U_z``)을 쓴다 — 두 루트의 ``target_names`` 가
      같은 문자열이 되도록 테스트로 고정한다.
    - 운전조건 μ 는 **노드 피처 브로드캐스트**(모든 노드에 동일 값 k 채널) —
      GeometryFNO 의 μ 채널 규약과 동형이다.

에지는 ``mesh.extract_all_edges()`` 로 뽑는다 (VTK 9.2+ 는 항상 점 전체 보존
모드) — 점 개수와 순서가 보존된다는 것이 계약이며(검사 강제), 정렬/비정렬
격자 무관하다.
"""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np
from numpy.typing import NDArray

from naviertwin.utils.logger import get_logger

logger = get_logger(__name__)

# case_tensorizer 와 동일한 성분 접미사 규칙 — 두 루트의 target_names 일치.
_COMPONENT_SUFFIXES = ("x", "y", "z")


def mesh_edge_index(mesh: Any) -> NDArray[np.int64]:
    """pyvista 메쉬에서 양방향 edge_index (2, 2E) 를 뽑는다.

    ``extract_all_edges()`` 는 점 개수·순서를 보존한 채 셀 에지를 라인
    셀로 돌려준다 — 이 보존이 깨지면 노드 피처와 에지가 어긋나므로
    검사로 강제한다.

    Args:
        mesh: pyvista DataSet (UnstructuredGrid/ImageData 등 VTK 계열).

    Returns:
        (2, 2E) int64 — 중복 제거된 양방향 에지.

    Raises:
        ValueError: 에지를 추출할 수 없거나 점 순서 보존이 깨진 경우.
    """
    # pyvista 0.47+/VTK 9.2+ 에서 use_all_points 는 항상 True (인자는 deprecated
    # — 명시하면 경고만 낸다). 점 보존 계약은 아래 검사가 강제한다.
    edges = mesh.extract_all_edges()
    if int(edges.n_points) != int(mesh.n_points):
        raise ValueError(
            "extract_all_edges 가 점 개수를 보존하지 않았습니다 "
            f"({edges.n_points} != {mesh.n_points}) — edge_index 를 만들 수 없습니다."
        )
    lines = np.asarray(edges.lines).reshape(-1, 3)
    if lines.size == 0:
        raise ValueError("메쉬에서 에지를 추출하지 못했습니다 (lines 비어 있음).")
    pairs = lines[:, 1:].astype(np.int64)  # (E, 2)
    both = np.concatenate([pairs, pairs[:, ::-1]], axis=0)  # 양방향화
    both = np.unique(both, axis=0)  # 중복 제거 (자기루프는 원래 없음)
    return np.ascontiguousarray(both.T)


def _field_channels(
    mesh: Any, name: str
) -> list[tuple[str, NDArray[np.float64]]]:
    """point_data 필드 하나를 (채널명, (N,) 배열) 목록으로 푼다.

    벡터 필드는 성분별 채널로 확장한다 — ``case_tensorizer._field_channels``
    와 같은 접미사 규칙(x/y/z, 4성분 이상은 인덱스).

    Raises:
        ValueError: 필드가 point_data 에 없는 경우.
    """
    if name not in mesh.point_data:
        available = list(mesh.point_data.keys())
        raise ValueError(
            f"필드 '{name}' 이(가) 메쉬 point_data 에 없습니다. 사용 가능: {available}"
        )
    values = np.asarray(mesh.point_data[name], dtype=np.float64)
    if values.ndim == 1:
        return [(name, values)]
    channels: list[tuple[str, NDArray[np.float64]]] = []
    for j in range(values.shape[1]):
        suffix = _COMPONENT_SUFFIXES[j] if values.shape[1] <= 3 else str(j)
        channels.append((f"{name}_{suffix}", values[:, j]))
    return channels


def graph_norm_from_cases(
    datasets: Sequence[Any],
    params: NDArray[np.float64],
    *,
    input_field_names: Sequence[str] = (),
) -> dict[str, Any]:
    """train 케이스들로 노드 피처 정규화 상수를 계산한다.

    group split 시 train-only 정규화 원칙(로드맵 검토 §6½ #2)을 지키기 위해
    빌더가 **train 케이스만** 넘겨 계산하고, 결과 dict 를 val/test 의
    :func:`case_to_graph` 에 그대로 주입한다.

    Returns:
        ``coord_center``/``coord_scale`` (3,), ``mu_center``/``mu_scale`` (k,),
        ``input_center``/``input_scale`` (n_inputs,) 를 담은 dict.
        폭이 0 인 축의 scale 은 1 로 보호된다.
    """
    points = np.vstack(
        [np.asarray(getattr(d, "mesh", d).points, dtype=np.float64)[:, :3] for d in datasets]
    )
    coord_center = points.mean(axis=0)
    coord_scale = points.std(axis=0)
    coord_scale = np.where(coord_scale > 0, coord_scale, 1.0)

    mu = np.asarray(params, dtype=np.float64)
    if mu.ndim == 1:
        mu = mu.reshape(-1, 1)
    mu_center = mu.mean(axis=0) if mu.size else np.zeros(mu.shape[1])
    mu_scale = mu.std(axis=0) if mu.size else np.ones(mu.shape[1])
    mu_scale = np.where(mu_scale > 0, mu_scale, 1.0)

    input_names = [str(n) for n in input_field_names]
    input_center = np.zeros(len(input_names))
    input_scale = np.ones(len(input_names))
    for j, name in enumerate(input_names):
        stacked = np.concatenate(
            [
                np.asarray(getattr(d, "mesh", d).point_data[name], dtype=np.float64).reshape(-1)
                for d in datasets
            ]
        )
        input_center[j] = float(stacked.mean())
        scale = float(stacked.std())
        input_scale[j] = scale if scale > 0 else 1.0

    return {
        "coord_center": coord_center,
        "coord_scale": coord_scale,
        "mu_center": mu_center,
        "mu_scale": mu_scale,
        "input_names": input_names,
        "input_center": input_center,
        "input_scale": input_scale,
    }


def case_to_graph(
    dataset: Any,
    mu: NDArray[np.float64],
    field_names: Sequence[str],
    *,
    input_field_names: Sequence[str] = (),
    norm: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """케이스 하나를 GNN 학습용 그래프 dict 로 바꾼다.

    노드 피처 ``x`` = [정규화 좌표 3성분 | 정규화 입력 필드 | 정규화 μ
    브로드캐스트]. 타깃 ``y`` = 요청 필드의 채널 전개(물리 단위 그대로 —
    타깃 표준화는 :class:`~naviertwin.core.gnn.gnn_surrogate.case_set_gnn.
    CaseSetGNN` 이 train 그래프만으로 내부 계산한다).

    Args:
        dataset: CFDDataset 또는 pyvista 메쉬.
        mu: (k,) 케이스 운전조건.
        field_names: 출력 필드 이름 목록 (벡터는 성분 전개).
        input_field_names: 노드 피처로 함께 넣을 point 필드
            (예: ``wall_distance``/``wall_sdf`` — ``attach_wall_features``
            산출물).
        norm: :func:`graph_norm_from_cases` 가 만든 정규화 상수. None 이면
            이 케이스 하나로 계산해 반환 dict 에 담는다.

    Returns:
        ``{"points", "edge_index", "edge_attr", "x", "y", "target_names",
        "norm"}`` — ``points`` (N, 3) float64 원 좌표, ``x`` (N, f) float32,
        ``y`` (N, C) float32, ``edge_attr`` (2E, dims+1) float32
        [Δ정규화좌표, ‖Δ‖].

    Raises:
        ValueError: 필드가 없거나 μ 차원이 norm 과 다른 경우.
    """
    mesh = getattr(dataset, "mesh", dataset)
    mu_arr = np.asarray(mu, dtype=np.float64).reshape(-1)
    if norm is None:
        norm = graph_norm_from_cases(
            [mesh], mu_arr.reshape(1, -1), input_field_names=input_field_names
        )
    if mu_arr.size != np.asarray(norm["mu_center"]).size:
        raise ValueError(
            f"μ 차원({mu_arr.size})이 정규화 상수 차원"
            f"({np.asarray(norm['mu_center']).size})과 다릅니다."
        )

    points = np.asarray(mesh.points, dtype=np.float64)[:, :3]
    edge_index = mesh_edge_index(mesh)

    coords_n = (points - norm["coord_center"]) / norm["coord_scale"]

    blocks: list[NDArray[np.float64]] = [coords_n]
    for j, name in enumerate([str(n) for n in input_field_names]):
        values = np.asarray(mesh.point_data[name], dtype=np.float64).reshape(-1, 1)
        blocks.append(
            (values - norm["input_center"][j]) / norm["input_scale"][j]
        )
    mu_n = (mu_arr - norm["mu_center"]) / norm["mu_scale"]
    if mu_n.size:
        blocks.append(np.broadcast_to(mu_n, (points.shape[0], mu_n.size)))
    x = np.hstack(blocks).astype(np.float32)

    # MGN 표준 상대좌표 에지 피처 (정규화 좌표 기준) — GCNConv 는 안 쓰지만
    # edge_attr 활용 모델(SAGE/NNConv, 후속)이 그대로 소비한다.
    delta = coords_n[edge_index[1]] - coords_n[edge_index[0]]
    edge_attr = np.concatenate(
        [delta, np.linalg.norm(delta, axis=1, keepdims=True)], axis=1
    ).astype(np.float32)

    target_names: list[str] = []
    y_channels: list[NDArray[np.float64]] = []
    for name in field_names:
        for channel_name, values in _field_channels(mesh, str(name)):
            target_names.append(channel_name)
            y_channels.append(values)
    if not y_channels:
        raise ValueError("출력 필드를 최소 1개 지정하세요.")
    y = np.stack(y_channels, axis=1).astype(np.float32)

    return {
        "points": points,
        "edge_index": edge_index,
        "edge_attr": edge_attr,
        "x": x,
        "y": y,
        "target_names": target_names,
        "norm": norm,
    }


__all__ = ["case_to_graph", "graph_norm_from_cases", "mesh_edge_index"]
