"""Canonical CFD project contracts.

The legacy :class:`CFDDataset` remains the reader-facing container.  These
contracts describe durable projects, case sets, fields, meshes, geometry,
snapshots, and boundary conditions without hiding required information in an
untyped metadata dictionary.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, Mapping, Sequence

SCHEMA_VERSION = "1.0"


class FieldLocation(str, Enum):
    """Where field samples live."""

    POINT = "point"
    CELL = "cell"
    FIELD = "field"


class FieldRole(str, Enum):
    """How a field participates in a learning problem."""

    STATE = "state"
    INPUT = "input"
    TARGET = "target"
    DERIVED = "derived"
    MASK = "mask"


class ConservationKind(str, Enum):
    """Interpolation/restriction semantics for a field."""

    UNKNOWN = "unknown"
    INTENSIVE = "intensive"
    EXTENSIVE = "extensive"
    CONSERVED = "conserved"
    CATEGORICAL = "categorical"


class BoundaryKind(str, Enum):
    """Canonical boundary classifications shared by all readers."""

    WALL = "wall"
    INLET = "inlet"
    OUTLET = "outlet"
    FARFIELD = "farfield"
    SYMMETRY = "symmetry"
    PERIODIC = "periodic"
    INTERFACE = "interface"
    UNKNOWN = "unknown"


class LineageKind(str, Enum):
    """Durable artifact categories produced by the twin workflow."""

    MAPPING = "mapping"
    MODEL = "model"
    PREDICTION = "prediction"
    VALIDATION = "validation"


def _require_id(value: str, label: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} must be a non-empty string")


def _duplicates(values: Sequence[str]) -> set[str]:
    seen: set[str] = set()
    duplicate: set[str] = set()
    for value in values:
        if value in seen:
            duplicate.add(value)
        seen.add(value)
    return duplicate


@dataclass(frozen=True)
class FieldDescriptor:
    """Physical field identity and array semantics."""

    name: str
    location: FieldLocation
    n_components: int = 1
    component_names: tuple[str, ...] = ()
    unit: str = ""
    dtype: str = ""
    role: FieldRole = FieldRole.STATE
    conservation: ConservationKind = ConservationKind.UNKNOWN
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _require_id(self.name, "field name")
        if self.n_components < 1:
            raise ValueError("field n_components must be at least 1")
        if self.component_names and len(self.component_names) != self.n_components:
            raise ValueError("component_names length must match n_components")
        if len(set(self.component_names)) != len(self.component_names):
            raise ValueError("component_names must be unique")


@dataclass(frozen=True)
class FieldDataRef:
    """Reference to one field array in canonical or tensor storage."""

    field_name: str
    storage_uri: str
    array_path: str
    shape: tuple[int, ...]
    time_index: int | None = None
    valid_mask_path: str | None = None
    checksum: str = ""

    def __post_init__(self) -> None:
        _require_id(self.field_name, "field reference name")
        _require_id(self.storage_uri, "field storage_uri")
        _require_id(self.array_path, "field array_path")
        if not self.shape or any(int(size) < 0 for size in self.shape):
            raise ValueError("field shape must contain non-negative dimensions")
        if self.time_index is not None and self.time_index < 0:
            raise ValueError("field time_index must be non-negative")


@dataclass(frozen=True)
class GeometryDescriptor:
    """Physical geometry identity, independent from mesh discretization."""

    geometry_id: str
    shape_signature: str = ""
    source_uri: str = ""
    sdf_field: str = ""
    wall_distance_field: str = ""
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _require_id(self.geometry_id, "geometry_id")


@dataclass(frozen=True)
class MeshDescriptor:
    """Discretization identity and dimensionality."""

    mesh_id: str
    geometry_id: str
    topology_hash: str
    coordinate_hash: str
    topological_dim: int
    embedding_dim: int
    n_points: int
    n_cells: int
    source_uri: str = ""
    coordinate_system: str = "cartesian"
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _require_id(self.mesh_id, "mesh_id")
        _require_id(self.geometry_id, "mesh geometry_id")
        if self.topological_dim not in (1, 2, 3):
            raise ValueError("topological_dim must be 1, 2, or 3")
        if self.embedding_dim not in (1, 2, 3):
            raise ValueError("embedding_dim must be 1, 2, or 3")
        if self.topological_dim > self.embedding_dim:
            raise ValueError("topological_dim cannot exceed embedding_dim")
        if self.n_points < 0 or self.n_cells < 0:
            raise ValueError("mesh sizes must be non-negative")


@dataclass(frozen=True)
class BoundaryPatch:
    """Stable boundary selection on one canonical mesh."""

    patch_id: str
    mesh_id: str
    name: str
    kind: BoundaryKind = BoundaryKind.UNKNOWN
    face_ids: tuple[int, ...] = ()
    face_ids_uri: str = ""
    source: str = "user"
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _require_id(self.patch_id, "patch_id")
        _require_id(self.mesh_id, "boundary mesh_id")
        _require_id(self.name, "boundary name")
        if any(face_id < 0 for face_id in self.face_ids):
            raise ValueError("boundary face_ids must be non-negative")
        if len(set(self.face_ids)) != len(self.face_ids):
            raise ValueError("boundary face_ids must be unique")


@dataclass(frozen=True)
class BoundaryCondition:
    """User/file supplied condition attached to a boundary patch."""

    patch_id: str
    kind: BoundaryKind
    values: Mapping[str, Any] = field(default_factory=dict)
    units: Mapping[str, str] = field(default_factory=dict)
    source: str = "user"
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _require_id(self.patch_id, "boundary condition patch_id")
        unknown_units = set(self.units) - set(self.values)
        if unknown_units:
            names = ", ".join(sorted(unknown_units))
            raise ValueError(f"boundary units have no matching values: {names}")


@dataclass(frozen=True)
class Snapshot:
    """One steady sample or one time slice of an unsteady case."""

    snapshot_id: str
    mesh_id: str
    fields: tuple[FieldDataRef, ...]
    time: float | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _require_id(self.snapshot_id, "snapshot_id")
        _require_id(self.mesh_id, "snapshot mesh_id")
        if self.time is not None and not math.isfinite(float(self.time)):
            raise ValueError("snapshot time must be finite")
        duplicates = _duplicates([item.field_name for item in self.fields])
        if duplicates:
            raise ValueError(f"duplicate snapshot fields: {sorted(duplicates)}")


@dataclass(frozen=True)
class Case:
    """A geometry/condition combination containing one or more snapshots."""

    case_id: str
    name: str
    parameters: Mapping[str, float]
    snapshots: tuple[Snapshot, ...]
    boundary_conditions: tuple[BoundaryCondition, ...] = ()
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _require_id(self.case_id, "case_id")
        _require_id(self.name, "case name")
        if not self.snapshots:
            raise ValueError("case must contain at least one snapshot")
        duplicate_ids = _duplicates([item.snapshot_id for item in self.snapshots])
        if duplicate_ids:
            raise ValueError(f"duplicate snapshot ids: {sorted(duplicate_ids)}")
        for name, value in self.parameters.items():
            _require_id(str(name), "parameter name")
            if not math.isfinite(float(value)):
                raise ValueError(f"case parameter {name!r} must be finite")
        times = [item.time for item in self.snapshots]
        if len(times) > 1 and any(value is None for value in times):
            raise ValueError("multi-snapshot cases require time on every snapshot")
        numeric_times = [float(value) for value in times if value is not None]
        if numeric_times != sorted(set(numeric_times)):
            raise ValueError("snapshot times must be unique and increasing")
        duplicate_patches = _duplicates(
            [item.patch_id for item in self.boundary_conditions]
        )
        if duplicate_patches:
            raise ValueError(
                f"duplicate boundary conditions for patches: {sorted(duplicate_patches)}"
            )


@dataclass(frozen=True)
class CaseSet:
    """Learning-ready collection spanning conditions, geometry, and time."""

    case_set_id: str
    name: str
    fields: tuple[FieldDescriptor, ...]
    cases: tuple[Case, ...]
    parameter_units: Mapping[str, str] = field(default_factory=dict)
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _require_id(self.case_set_id, "case_set_id")
        _require_id(self.name, "case set name")
        if not self.cases:
            raise ValueError("case set must contain at least one case")
        duplicate_cases = _duplicates([item.case_id for item in self.cases])
        if duplicate_cases:
            raise ValueError(f"duplicate case ids: {sorted(duplicate_cases)}")
        duplicate_fields = _duplicates([item.name for item in self.fields])
        if duplicate_fields:
            raise ValueError(f"duplicate field descriptors: {sorted(duplicate_fields)}")
        parameter_names = set(self.cases[0].parameters)
        for case in self.cases[1:]:
            if set(case.parameters) != parameter_names:
                raise ValueError("all cases in a case set must use the same parameters")
        if set(self.parameter_units) - parameter_names:
            raise ValueError("parameter_units contains unknown parameters")


@dataclass(frozen=True)
class LineageRecord:
    """Provenance record for one transformation or generated artifact."""

    artifact_id: str
    kind: LineageKind
    input_ids: tuple[str, ...] = ()
    artifact_uri: str = ""
    checksum: str = ""
    strategy: str = ""
    created_at: str = ""
    parameters: Mapping[str, Any] = field(default_factory=dict)
    metrics: Mapping[str, float] = field(default_factory=dict)
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _require_id(self.artifact_id, "lineage artifact_id")
        if any(not str(value).strip() for value in self.input_ids):
            raise ValueError("lineage input_ids must be non-empty strings")
        for name, value in self.metrics.items():
            _require_id(str(name), "lineage metric name")
            if not math.isfinite(float(value)):
                raise ValueError(f"lineage metric {name!r} must be finite")


@dataclass(frozen=True)
class TwinProject:
    """Top-level durable NavierTwin project manifest."""

    project_id: str
    name: str
    case_sets: tuple[CaseSet, ...]
    meshes: tuple[MeshDescriptor, ...]
    geometries: tuple[GeometryDescriptor, ...]
    boundaries: tuple[BoundaryPatch, ...] = ()
    lineage: tuple[LineageRecord, ...] = ()
    schema_version: str = SCHEMA_VERSION
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _require_id(self.project_id, "project_id")
        _require_id(self.name, "project name")
        if self.schema_version != SCHEMA_VERSION:
            raise ValueError(
                f"unsupported project schema {self.schema_version!r}; expected {SCHEMA_VERSION!r}"
            )
        mesh_ids = [item.mesh_id for item in self.meshes]
        geometry_ids = [item.geometry_id for item in self.geometries]
        patch_ids = [item.patch_id for item in self.boundaries]
        case_set_ids = [item.case_set_id for item in self.case_sets]
        lineage_ids = [item.artifact_id for item in self.lineage]
        for label, values in (
            ("mesh", mesh_ids),
            ("geometry", geometry_ids),
            ("boundary patch", patch_ids),
            ("case set", case_set_ids),
            ("lineage artifact", lineage_ids),
        ):
            duplicate = _duplicates(values)
            if duplicate:
                raise ValueError(f"duplicate {label} ids: {sorted(duplicate)}")

        mesh_id_set = set(mesh_ids)
        geometry_id_set = set(geometry_ids)
        patch_id_set = set(patch_ids)
        for mesh in self.meshes:
            if mesh.geometry_id not in geometry_id_set:
                raise ValueError(f"mesh {mesh.mesh_id!r} references unknown geometry")
        for patch in self.boundaries:
            if patch.mesh_id not in mesh_id_set:
                raise ValueError(f"boundary {patch.patch_id!r} references unknown mesh")
        for case_set in self.case_sets:
            field_names = {item.name for item in case_set.fields}
            for case in case_set.cases:
                for condition in case.boundary_conditions:
                    if condition.patch_id not in patch_id_set:
                        raise ValueError(
                            f"case {case.case_id!r} references unknown boundary patch "
                            f"{condition.patch_id!r}"
                        )
                for snapshot in case.snapshots:
                    if snapshot.mesh_id not in mesh_id_set:
                        raise ValueError(
                            f"snapshot {snapshot.snapshot_id!r} references unknown mesh"
                        )
                    unknown_fields = {
                        item.field_name for item in snapshot.fields
                    } - field_names
                    if unknown_fields:
                        raise ValueError(
                            f"snapshot {snapshot.snapshot_id!r} references unknown fields: "
                            f"{sorted(unknown_fields)}"
                        )

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-safe nested representation."""

        return asdict(self)


def _field_descriptor_from_dict(data: Mapping[str, Any]) -> FieldDescriptor:
    return FieldDescriptor(
        name=str(data["name"]),
        location=FieldLocation(data["location"]),
        n_components=int(data.get("n_components", 1)),
        component_names=tuple(data.get("component_names", ())),
        unit=str(data.get("unit", "")),
        dtype=str(data.get("dtype", "")),
        role=FieldRole(data.get("role", FieldRole.STATE)),
        conservation=ConservationKind(
            data.get("conservation", ConservationKind.UNKNOWN)
        ),
        metadata=dict(data.get("metadata", {})),
    )


def _field_ref_from_dict(data: Mapping[str, Any]) -> FieldDataRef:
    return FieldDataRef(
        field_name=str(data["field_name"]),
        storage_uri=str(data["storage_uri"]),
        array_path=str(data["array_path"]),
        shape=tuple(int(value) for value in data["shape"]),
        time_index=(
            None if data.get("time_index") is None else int(data["time_index"])
        ),
        valid_mask_path=data.get("valid_mask_path"),
        checksum=str(data.get("checksum", "")),
    )


def project_from_dict(data: Mapping[str, Any]) -> TwinProject:
    """Build and validate a project from decoded JSON data."""

    geometries = tuple(
        GeometryDescriptor(
            geometry_id=str(item["geometry_id"]),
            shape_signature=str(item.get("shape_signature", "")),
            source_uri=str(item.get("source_uri", "")),
            sdf_field=str(item.get("sdf_field", "")),
            wall_distance_field=str(item.get("wall_distance_field", "")),
            metadata=dict(item.get("metadata", {})),
        )
        for item in data.get("geometries", ())
    )
    meshes = tuple(
        MeshDescriptor(
            mesh_id=str(item["mesh_id"]),
            geometry_id=str(item["geometry_id"]),
            topology_hash=str(item.get("topology_hash", "")),
            coordinate_hash=str(item.get("coordinate_hash", "")),
            topological_dim=int(item["topological_dim"]),
            embedding_dim=int(item["embedding_dim"]),
            n_points=int(item["n_points"]),
            n_cells=int(item["n_cells"]),
            source_uri=str(item.get("source_uri", "")),
            coordinate_system=str(item.get("coordinate_system", "cartesian")),
            metadata=dict(item.get("metadata", {})),
        )
        for item in data.get("meshes", ())
    )
    boundaries = tuple(
        BoundaryPatch(
            patch_id=str(item["patch_id"]),
            mesh_id=str(item["mesh_id"]),
            name=str(item["name"]),
            kind=BoundaryKind(item.get("kind", BoundaryKind.UNKNOWN)),
            face_ids=tuple(int(value) for value in item.get("face_ids", ())),
            face_ids_uri=str(item.get("face_ids_uri", "")),
            source=str(item.get("source", "user")),
            metadata=dict(item.get("metadata", {})),
        )
        for item in data.get("boundaries", ())
    )
    lineage = tuple(
        LineageRecord(
            artifact_id=str(item["artifact_id"]),
            kind=LineageKind(item["kind"]),
            input_ids=tuple(str(value) for value in item.get("input_ids", ())),
            artifact_uri=str(item.get("artifact_uri", "")),
            checksum=str(item.get("checksum", "")),
            strategy=str(item.get("strategy", "")),
            created_at=str(item.get("created_at", "")),
            parameters=dict(item.get("parameters", {})),
            metrics={
                str(key): float(value)
                for key, value in item.get("metrics", {}).items()
            },
            metadata=dict(item.get("metadata", {})),
        )
        for item in data.get("lineage", ())
    )
    case_sets: list[CaseSet] = []
    for raw_set in data.get("case_sets", ()):
        fields = tuple(
            _field_descriptor_from_dict(item) for item in raw_set.get("fields", ())
        )
        cases: list[Case] = []
        for raw_case in raw_set.get("cases", ()):
            snapshots = tuple(
                Snapshot(
                    snapshot_id=str(item["snapshot_id"]),
                    mesh_id=str(item["mesh_id"]),
                    time=(None if item.get("time") is None else float(item["time"])),
                    fields=tuple(
                        _field_ref_from_dict(field_item)
                        for field_item in item.get("fields", ())
                    ),
                    metadata=dict(item.get("metadata", {})),
                )
                for item in raw_case.get("snapshots", ())
            )
            conditions = tuple(
                BoundaryCondition(
                    patch_id=str(item["patch_id"]),
                    kind=BoundaryKind(item.get("kind", BoundaryKind.UNKNOWN)),
                    values=dict(item.get("values", {})),
                    units=dict(item.get("units", {})),
                    source=str(item.get("source", "user")),
                    metadata=dict(item.get("metadata", {})),
                )
                for item in raw_case.get("boundary_conditions", ())
            )
            cases.append(
                Case(
                    case_id=str(raw_case["case_id"]),
                    name=str(raw_case["name"]),
                    parameters={
                        str(key): float(value)
                        for key, value in raw_case.get("parameters", {}).items()
                    },
                    snapshots=snapshots,
                    boundary_conditions=conditions,
                    metadata=dict(raw_case.get("metadata", {})),
                )
            )
        case_sets.append(
            CaseSet(
                case_set_id=str(raw_set["case_set_id"]),
                name=str(raw_set["name"]),
                fields=fields,
                cases=tuple(cases),
                parameter_units=dict(raw_set.get("parameter_units", {})),
                metadata=dict(raw_set.get("metadata", {})),
            )
        )
    return TwinProject(
        project_id=str(data["project_id"]),
        name=str(data["name"]),
        case_sets=tuple(case_sets),
        meshes=meshes,
        geometries=geometries,
        boundaries=boundaries,
        lineage=lineage,
        schema_version=str(data.get("schema_version", SCHEMA_VERSION)),
        metadata=dict(data.get("metadata", {})),
    )


__all__ = [
    "SCHEMA_VERSION",
    "BoundaryCondition",
    "BoundaryKind",
    "BoundaryPatch",
    "Case",
    "CaseSet",
    "ConservationKind",
    "FieldDataRef",
    "FieldDescriptor",
    "FieldLocation",
    "FieldRole",
    "GeometryDescriptor",
    "LineageKind",
    "LineageRecord",
    "MeshDescriptor",
    "Snapshot",
    "TwinProject",
    "project_from_dict",
]
