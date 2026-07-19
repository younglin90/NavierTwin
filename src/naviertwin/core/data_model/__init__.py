"""캐노니컬 데이터 모델 패키지 (canonical CFD data model, 로드맵 §6½).

케이스/데이터셋을 좌표 전수 비교 없이 식별·비교하기 위한 해시 기반
시그니처를 제공한다. 자세한 설계 근거는 :mod:`.signature` 참고.
"""

from naviertwin.core.data_model.adapters import case_set_from_cfd_datasets
from naviertwin.core.data_model.ingest import (
    SOURCE_MANIFEST_SCHEMA,
    CaseSetSource,
    CaseSource,
    ProjectSource,
    ingest_project_source,
    ingest_project_source_manifest,
    load_project_source_manifest,
    project_source_from_dict,
    save_project_source_manifest,
)
from naviertwin.core.data_model.manifest import (
    load_project_manifest,
    save_project_manifest,
)
from naviertwin.core.data_model.schema import (
    SCHEMA_VERSION,
    BoundaryCondition,
    BoundaryKind,
    BoundaryPatch,
    Case,
    CaseSet,
    ConservationKind,
    FieldDataRef,
    FieldDescriptor,
    FieldLocation,
    FieldRole,
    GeometryDescriptor,
    LineageKind,
    LineageRecord,
    MeshDescriptor,
    Snapshot,
    TwinProject,
    project_from_dict,
)
from naviertwin.core.data_model.signature import (
    DatasetSignature,
    assign_geometry_ids,
    compute_signature,
    same_mesh,
)
from naviertwin.core.data_model.workspace import TwinWorkspace, WorkspaceStatus

__all__ = [
    "DatasetSignature",
    "assign_geometry_ids",
    "compute_signature",
    "same_mesh",
    "SCHEMA_VERSION",
    "SOURCE_MANIFEST_SCHEMA",
    "BoundaryCondition",
    "BoundaryKind",
    "BoundaryPatch",
    "Case",
    "CaseSet",
    "CaseSetSource",
    "CaseSource",
    "ConservationKind",
    "FieldDataRef",
    "FieldDescriptor",
    "FieldLocation",
    "FieldRole",
    "GeometryDescriptor",
    "LineageKind",
    "LineageRecord",
    "MeshDescriptor",
    "ProjectSource",
    "Snapshot",
    "TwinProject",
    "TwinWorkspace",
    "WorkspaceStatus",
    "case_set_from_cfd_datasets",
    "ingest_project_source",
    "ingest_project_source_manifest",
    "load_project_manifest",
    "load_project_source_manifest",
    "project_from_dict",
    "project_source_from_dict",
    "save_project_manifest",
    "save_project_source_manifest",
]
