"""GINO(Graph-Informed Neural Operator) — 케이스 세트 점군(point-cloud) 래퍼.

Route 2(메쉬 네이티브) 두 번째 배선. ``mesh_gnn``(GCN, 메쉬 그래프)과 달리
``neuraloperator`` 의 ``GINO`` 를 감싼다 — 입력/출력 모두 임의 점군에서
동작하고(고정 edge_index 불필요), 내부적으로 균일 잠재 격자(latent grid) 위
FNO 로 전역 상호작용을 계산한다(Li et al., 2023).
"""

from naviertwin.core.operator_learning.gino.gino_wrapper import (
    GINOCaseSetOperator,
    case_to_pointcloud,
    pointcloud_norm_from_cases,
)

__all__ = [
    "GINOCaseSetOperator",
    "case_to_pointcloud",
    "pointcloud_norm_from_cases",
]
