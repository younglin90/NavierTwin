"""단일 케이스 시계열 → :class:`MeshGraphNets` 트레젝토리 빌더.

``case_set_mgn.py``(:class:`~naviertwin.core.gnn.meshgraphnets.case_set_mgn.
CaseSetMGN`, 전략 키 ``mesh_gnn_mp``)는 정상(steady) 케이스 세트를 "1스텝
가짜 트레젝토리"로 재해석해 파라미터→필드 회귀를 하는 것이지, 진짜 시간
롤아웃이 아니다. 이 모듈은 그 반대 — **진짜 자기회귀 시간 롤아웃**을 지원하는
원본 :class:`~naviertwin.core.gnn.meshgraphnets.meshgraphnets.MeshGraphNets`
가 요구하는 ``dataset={"trajectories", "edge_index", "edge_features"}`` 형태를,
단일 케이스(같은 메쉬, 여러 타임스텝 스냅샷)에서 조립한다.

메쉬는 시간에 따라 바뀌지 않으므로(같은 케이스), 그래프(에지 연결·에지
피처)는 **한 번만** 계산하고, 노드 상태만 타임스텝마다 쌓는다 — 노드 상태에
좌표를 포함하지 않는다(에지 피처가 이미 상대 위치 Δ좌표를 담아 기하 정보를
전달하므로, MeshGraphNets 원 논문의 설계와 같다). ``case_set_mgn`` 의
[정적 피처 | 타깃] 이어붙이기 트릭은 여기선 필요 없다 — 트레젝토리 상태
자체가 곧 예측 대상 필드다.

빌더는 기존 :func:`~naviertwin.core.gnn.case_graph.case_to_graph` /
:func:`~naviertwin.core.gnn.case_graph.mesh_edge_index` (에지 추출·정규화)와
:func:`~naviertwin.core.preprocessing.expand_unsteady_case_snapshots`
(다중 타임스텝 벡터 필드를 스텝별 메쉬로 materialize)를 그대로 재사용한다 —
새 그래프 추출 로직을 만들지 않는다.
"""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np

from naviertwin.utils.logger import get_logger

logger = get_logger(__name__)


def case_to_rollout_trajectory(
    dataset: Any,
    field_names: Sequence[str],
) -> dict[str, Any]:
    """단일 케이스(다중 타임스텝)를 :meth:`MeshGraphNets.fit` 트레젝토리로 바꾼다.

    Args:
        dataset: 다중 타임스텝 ``CFDDataset`` (같은 메쉬, 시간에 따라 필드만
            변한다). ``metadata["time_series_fields"]`` 가 있으면 그걸 우선
            쓰고, 없으면 ``extract_field_snapshots`` 로 폴백한다(둘 다
            :func:`~naviertwin.core.preprocessing.expand_unsteady_case_snapshots`
            가 처리).
        field_names: 롤아웃 대상 필드 — 문자열 목록. 벡터 필드는
            :func:`~naviertwin.core.gnn.case_graph.case_to_graph` 와 같은
            규칙으로 성분 채널(``U_x`` 등)로 전개된다.

    Returns:
        ``{"trajectories": (1, T+1, N, C) float32, "edge_index": (2, 2E)
        int64, "edge_features": (2E, e) float32, "target_names": list[str],
        "times": list[float] (길이 T+1, 오름차순), "points": (N, 3) float64}``.

    Raises:
        ValueError: 필드 미지정, 타임스텝 3개 미만(다음스텝 예측 학습에
            최소 2쌍 필요), 또는 타임스텝마다 타깃 채널 구성이 다른 경우.
    """
    from naviertwin.core.gnn.case_graph import case_to_graph, graph_norm_from_cases
    from naviertwin.core.preprocessing import expand_unsteady_case_snapshots

    fields = [str(f) for f in field_names if str(f).strip()]
    if not fields:
        raise ValueError("트레젝토리를 만들 출력 필드를 최소 1개 지정하세요.")

    n_steps_declared = int(getattr(dataset, "n_time_steps", 0))
    if n_steps_declared < 3:
        raise ValueError(
            "MeshGraphNets 롤아웃 학습에는 케이스당 타임스텝이 최소 3개 필요합니다"
            f" (다음스텝 예측 쌍이 최소 2개 필요) — 현재: {n_steps_declared}."
        )

    # 케이스 하나짜리 "케이스 세트" 로 취급해 기존 다중 타임스텝 전개 배관을
    # 재사용한다 — μ 는 없으므로(k=0) 시간만 전개된다.
    empty_params = np.zeros((1, 0), dtype=np.float64)
    snapshots, _params, _names, has_time = expand_unsteady_case_snapshots(
        [dataset], empty_params, [], field_names=fields
    )
    if not has_time or len(snapshots) < 3:
        raise ValueError(
            "시계열 데이터가 아니거나(타임스텝 1개) 전개된 스냅샷이 부족합니다"
            f" (현재: {len(snapshots)})."
        )

    # 메쉬가 전 스텝에서 동일하므로 정규화 상수·에지는 t=0 스냅샷으로 한 번만
    # 계산하고 모든 스텝에 그대로 재사용한다(케이스 간 정규화와 같은 원칙).
    norm = graph_norm_from_cases([snapshots[0]], empty_params)
    graphs = [
        case_to_graph(snap, np.zeros(0), fields, norm=norm) for snap in snapshots
    ]

    target_names = list(graphs[0]["target_names"])
    for step_idx, graph in enumerate(graphs[1:], start=1):
        if list(graph["target_names"]) != target_names:
            raise ValueError(
                f"타임스텝 {step_idx} 의 타깃 채널({graph['target_names']})이 "
                f"t=0({target_names})과 다릅니다 — 시계열 내내 같은 필드 구성이어야 "
                "합니다."
            )

    trajectory = np.stack([g["y"] for g in graphs], axis=0)[np.newaxis, ...]
    trajectory = trajectory.astype(np.float32)

    times = [float(snap.metadata.get("source_time", i)) for i, snap in enumerate(snapshots)]

    return {
        "trajectories": trajectory,
        "edge_index": graphs[0]["edge_index"],
        "edge_features": graphs[0]["edge_attr"],
        "target_names": target_names,
        "times": times,
        "points": graphs[0]["points"],
    }


__all__ = ["case_to_rollout_trajectory"]
