"""데이터셋 시그니처 — 메쉬 위상/좌표 해시 기반 케이스 식별자.

외부 검토("canonical CFD data model", 로드맵 §6½)가 요구한 최소 단위. 케이스
세트에서 두 가지 질문을 좌표 배열 전수 비교 O(n_points) 대신 해시 비교 O(1)
로 판정할 수 있게 한다:

    1. **"같은 격자인가?"** — topology_hash + coordinate_hash 가 모두 같으면
       같은 격자다 → 스냅샷 행렬을 쌓을 수 있으므로 ROM(POD 등) 사용 가능.
    2. **"같은 형상인가?"** — 시그니처가 같은 케이스에 같은 정수 geometry_id
       를 부여(:func:`assign_geometry_ids`)해 그룹 스플릿의 ``group_ids`` 로
       바로 쓴다 → 같은 형상이 train/test 를 가로지르는 누수를 막는다.

설계 원칙:
    - **반올림 금지**: 좌표는 float64 바이트를 그대로 해시한다. 반올림(예:
      1e-9 격자 스냅)을 넣으면 반올림 경계에 걸린 좌표가 "거의 같은데 다른
      해시"를 만들어 판정이 비결정적으로 흔들린다. 부동소수점이 1 ULP 라도
      다르면 다른 격자로 본다 — 보수적 판정이다(같은 격자를 다르다고 하는
      false negative 는 성능 손해로 그치지만, 다른 격자를 같다고 하는 false
      positive 는 스냅샷 행렬을 오염시킨다).
    - core 는 web/Qt 에 의존하지 않는다 (아키텍처 규칙). 입력은 duck typing
      으로 받는다 — ``.mesh`` 속성이 있으면 CFDDataset 로 간주한다.

향후 통합 지점:
    :func:`naviertwin.core.digital_twin.strategies._same_mesh_points` 는 현재
    좌표 배열 전수 비교(``np.allclose``)로 같은 판정을 수행한다 — 이 모듈의
    :func:`same_mesh` 로 대체하면 케이스 수 × 점 수에 비례하던 비교 비용이
    케이스당 해시 1회 + 문자열 비교로 줄어든다. (해당 파일은 병행 작업 중이라
    이번 변경에서는 손대지 않고, 통합 지점만 여기에 기록해 둔다. 단,
    ``np.allclose`` 는 허용오차 비교이고 해시는 완전 일치 비교라는 의미 차이
    가 있으므로 교체 시 보수화됨을 감안할 것.)
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np

__all__ = [
    "DatasetSignature",
    "compute_signature",
    "same_mesh",
    "assign_geometry_ids",
]

# sha256 전체(64 hex) 대신 앞 16 hex(= 64 bit)만 쓰는 근거:
#   - 충돌 확률: 생일 역설 기준 64 bit 해시는 약 5×10^9 개를 모아야 충돌 확률
#     50% 에 도달한다. 실제 케이스 세트는 많아야 수천~수만 개 → 우연 충돌
#     확률 < 1e-10, 사실상 0 이다.
#   - 로그 가독성: 16자는 로그/UI 한 줄에서 눈으로 비교 가능하다. 64자는
#     줄을 넘겨 오히려 비교 실수를 유발한다.
_HASH_HEX_LEN = 16


@dataclass(frozen=True)
class DatasetSignature:
    """메쉬의 위상/좌표 해시 기반 데이터셋 식별자.

    두 시그니처의 ``topology_hash`` 와 ``coordinate_hash`` 가 모두 같으면
    같은 격자(같은 연결성 + 같은 좌표)로 판정한다 — :func:`same_mesh`.

    Attributes:
        topology_hash: 셀 연결성 해시 — UnstructuredGrid 의 ``cells`` 배열과
            ``celltypes`` 배열을 이어 붙인 바이트의 sha256 앞 16자리(hex).
            좌표가 이동해도 연결성이 같으면 유지된다 (예: 강체 이동한 격자).
        coordinate_hash: 점 좌표 해시 — ``points`` 를 float64 로 정규화한
            바이트의 sha256 앞 16자리(hex). **반올림하지 않는다** — 1 ULP
            차이도 다른 해시가 된다 (모듈 docstring 의 설계 원칙 참고).
        n_points: 점 개수 (사람용 요약 — 해시 없이도 빠른 1차 비교 가능).
        n_cells: 셀 개수 (동일).
    """

    topology_hash: str
    coordinate_hash: str
    n_points: int
    n_cells: int


def _as_unstructured(mesh_or_dataset: Any) -> Any:
    """CFDDataset(.mesh) 또는 pyvista 메쉬를 UnstructuredGrid 로 정규화한다."""
    import pyvista as pv

    mesh = getattr(mesh_or_dataset, "mesh", mesh_or_dataset)
    if mesh is None:
        raise TypeError("mesh 가 None 입니다 — 시그니처를 계산할 수 없습니다.")
    if not isinstance(mesh, pv.UnstructuredGrid):
        # ImageData/StructuredGrid 등은 cells/celltypes 배열이 없다 — 내부
        # 표준 표현(UnstructuredGrid)으로 캐스팅 후 계산한다.
        mesh = mesh.cast_to_unstructured_grid()
    return mesh


def compute_signature(mesh_or_dataset: Any) -> DatasetSignature:
    """메쉬(또는 CFDDataset)의 :class:`DatasetSignature` 를 계산한다.

    Args:
        mesh_or_dataset: ``.mesh`` 속성을 가진 CFDDataset, 또는 pyvista 메쉬
            자체. UnstructuredGrid 가 아니면 ``cast_to_unstructured_grid()``
            후 계산한다.

    Returns:
        :class:`DatasetSignature`.

    Raises:
        TypeError: mesh 가 None 인 경우.
    """
    mesh = _as_unstructured(mesh_or_dataset)

    # dtype/메모리 배치 정규화: 같은 격자가 float32/float64, C/F-order 로
    # 저장 방식만 다를 때 해시가 갈리지 않도록 float64 C-연속으로 통일한다.
    # (float32→float64 캐스팅은 값 손실이 없는 확대 변환 — 반올림이 아니다.)
    points = np.ascontiguousarray(np.asarray(mesh.points, dtype=np.float64))
    cells = np.ascontiguousarray(np.asarray(mesh.cells, dtype=np.int64))
    celltypes = np.ascontiguousarray(np.asarray(mesh.celltypes, dtype=np.int64))

    topo = hashlib.sha256()
    topo.update(cells.tobytes())
    topo.update(b"|")  # cells/celltypes 경계 표식 — 연접 모호성 방지
    topo.update(celltypes.tobytes())

    coord = hashlib.sha256(points.tobytes())

    return DatasetSignature(
        topology_hash=topo.hexdigest()[:_HASH_HEX_LEN],
        coordinate_hash=coord.hexdigest()[:_HASH_HEX_LEN],
        n_points=int(mesh.n_points),
        n_cells=int(mesh.n_cells),
    )


def same_mesh(a: Any, b: Any) -> bool:
    """두 입력이 같은 격자인지 판정한다 (topology + coordinate 해시 모두 일치).

    Args:
        a: :class:`DatasetSignature`, CFDDataset, 또는 pyvista 메쉬.
        b: 동일.

    Returns:
        topology_hash 와 coordinate_hash 가 모두 같으면 True.
    """
    sig_a = a if isinstance(a, DatasetSignature) else compute_signature(a)
    sig_b = b if isinstance(b, DatasetSignature) else compute_signature(b)
    return (
        sig_a.topology_hash == sig_b.topology_hash
        and sig_a.coordinate_hash == sig_b.coordinate_hash
    )


def assign_geometry_ids(datasets: Sequence[Any]) -> list[int]:
    """시그니처가 같은 케이스에 같은 정수 geometry_id 를 부여한다.

    (topology_hash, coordinate_hash) 쌍이 같은 케이스는 같은 id 를 받고,
    id 는 **등장 순서대로** 0, 1, 2, ... 로 매겨진다 — 같은 입력 순서는 항상
    같은 결과를 재현한다.

    반환값은 그대로 그룹 인지 분할/학습 API 에 넣을 수 있는 값이다:

        - :func:`naviertwin.core.preprocessing.group_split.group_train_val_test_split`
          의 ``group_ids=`` — 같은 형상이 train/val/test 를 가로지르지 않게
          하는 geometry 단위 분할.
        - :func:`naviertwin.web.service.build_geometry_fno_twin` 의
          ``group_ids=`` — GeometryFNO 학습 시 형상 단위 스플릿.

    Args:
        datasets: CFDDataset 또는 pyvista 메쉬의 목록.

    Returns:
        케이스별 geometry_id (길이 ``len(datasets)``, 0 부터 시작).
    """
    seen: dict[tuple[str, str], int] = {}
    ids: list[int] = []
    for ds in datasets:
        sig = compute_signature(ds)
        key = (sig.topology_hash, sig.coordinate_hash)
        if key not in seen:
            seen[key] = len(seen)
        ids.append(seen[key])
    return ids
