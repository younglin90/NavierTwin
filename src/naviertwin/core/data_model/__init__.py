"""캐노니컬 데이터 모델 패키지 (canonical CFD data model, 로드맵 §6½).

케이스/데이터셋을 좌표 전수 비교 없이 식별·비교하기 위한 해시 기반
시그니처를 제공한다. 자세한 설계 근거는 :mod:`.signature` 참고.
"""

from naviertwin.core.data_model.signature import (
    DatasetSignature,
    assign_geometry_ids,
    compute_signature,
    same_mesh,
)

__all__ = [
    "DatasetSignature",
    "assign_geometry_ids",
    "compute_signature",
    "same_mesh",
]
