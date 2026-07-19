"""Source-manifest driven CFD project ingestion."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence, TypeVar

from naviertwin.core.data_model.adapters import case_set_from_cfd_datasets
from naviertwin.core.data_model.schema import (
    SCHEMA_VERSION,
    BoundaryPatch,
    CaseSet,
    GeometryDescriptor,
    MeshDescriptor,
    TwinProject,
)
from naviertwin.utils.atomic_io import atomic_write_text
from naviertwin.utils.json_safe import safe_dumps

SOURCE_MANIFEST_SCHEMA = "naviertwin-source-1.0"


def _require_text(value: str, label: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} must be a non-empty string")


@dataclass(frozen=True)
class CaseSource:
    """One CFD source and its user-supplied physical coordinates."""

    case_id: str
    name: str
    path: str
    parameters: Mapping[str, float] = field(default_factory=dict)
    geometry_id: str = ""

    def __post_init__(self) -> None:
        _require_text(self.case_id, "source case_id")
        _require_text(self.name, "source case name")
        _require_text(self.path, "source path")


@dataclass(frozen=True)
class CaseSetSource:
    """Source files forming one learning problem."""

    case_set_id: str
    name: str
    cases: tuple[CaseSource, ...]
    parameter_units: Mapping[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _require_text(self.case_set_id, "source case_set_id")
        _require_text(self.name, "source case set name")
        if not self.cases:
            raise ValueError("source case set must contain at least one case")
        ids = [item.case_id for item in self.cases]
        if len(ids) != len(set(ids)):
            raise ValueError("source case ids must be unique within a case set")
        parameter_names = set(self.cases[0].parameters)
        for case in self.cases[1:]:
            if set(case.parameters) != parameter_names:
                raise ValueError("all source cases must use the same parameter axes")
        if set(self.parameter_units) - parameter_names:
            raise ValueError("source parameter_units contains unknown parameters")


@dataclass(frozen=True)
class ProjectSource:
    """Portable list of CFD inputs used to build a canonical project."""

    project_id: str
    name: str
    case_sets: tuple[CaseSetSource, ...]
    schema: str = SOURCE_MANIFEST_SCHEMA
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _require_text(self.project_id, "source project_id")
        _require_text(self.name, "source project name")
        if self.schema != SOURCE_MANIFEST_SCHEMA:
            raise ValueError(f"unsupported source manifest schema: {self.schema!r}")
        if not self.case_sets:
            raise ValueError("source project must contain at least one case set")
        ids = [item.case_set_id for item in self.case_sets]
        if len(ids) != len(set(ids)):
            raise ValueError("source case set ids must be unique")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def project_source_from_dict(data: Mapping[str, Any]) -> ProjectSource:
    """Decode and validate a source manifest."""

    case_sets: list[CaseSetSource] = []
    for raw_set in data.get("case_sets", ()):
        cases = tuple(
            CaseSource(
                case_id=str(item["case_id"]),
                name=str(item.get("name", item["case_id"])),
                path=str(item["path"]),
                parameters={
                    str(key): float(value)
                    for key, value in item.get("parameters", {}).items()
                },
                geometry_id=str(item.get("geometry_id", "")),
            )
            for item in raw_set.get("cases", ())
        )
        case_sets.append(
            CaseSetSource(
                case_set_id=str(raw_set["case_set_id"]),
                name=str(raw_set.get("name", raw_set["case_set_id"])),
                cases=cases,
                parameter_units=dict(raw_set.get("parameter_units", {})),
            )
        )
    return ProjectSource(
        project_id=str(data["project_id"]),
        name=str(data.get("name", data["project_id"])),
        case_sets=tuple(case_sets),
        schema=str(data.get("schema", SOURCE_MANIFEST_SCHEMA)),
        metadata=dict(data.get("metadata", {})),
    )


def save_project_source_manifest(source: ProjectSource, path: str | Path) -> Path:
    """Atomically save a portable source manifest."""

    return atomic_write_text(path, safe_dumps(source.to_dict(), sort_keys=True) + "\n")


def load_project_source_manifest(path: str | Path) -> ProjectSource:
    """Load and validate a portable source manifest."""

    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("source manifest root must be an object")
    return project_source_from_dict(payload)


T = TypeVar("T")


def _merge_unique(
    target: dict[str, T], values: Sequence[T], key: Callable[[T], str], label: str
) -> None:
    for value in values:
        item_id = key(value)
        existing = target.get(item_id)
        if existing is not None and existing != value:
            raise ValueError(f"conflicting {label} definition for {item_id!r}")
        target[item_id] = value


def ingest_project_source(
    source: ProjectSource,
    *,
    base_dir: str | Path = ".",
    reader_fn: Callable[[Path], Any] | None = None,
    use_canonical_cache: bool = False,
    canonical_cache_dir: str | Path | None = None,
) -> TwinProject:
    """Read all source cases and build a validated canonical project graph."""

    if reader_fn is None:
        from naviertwin.core.cfd_reader import ReaderFactory

        reader_fn = ReaderFactory.create_and_read
    cache = None
    if use_canonical_cache:
        from naviertwin.core.storage.canonical_cache import CanonicalCache

        cache = CanonicalCache(
            None if canonical_cache_dir is None else Path(canonical_cache_dir)
        )

    root = Path(base_dir).expanduser().resolve()
    case_sets: list[CaseSet] = []
    meshes: dict[str, MeshDescriptor] = {}
    geometries: dict[str, GeometryDescriptor] = {}
    boundaries: dict[str, BoundaryPatch] = {}
    for source_set in source.case_sets:
        paths: list[Path] = []
        datasets: list[Any] = []
        for case in source_set.cases:
            path = Path(case.path).expanduser()
            if not path.is_absolute():
                path = root / path
            path = path.resolve()
            if not path.exists():
                raise FileNotFoundError(f"CFD case source does not exist: {path}")
            paths.append(path)
            if cache is None:
                datasets.append(reader_fn(path))
            else:
                datasets.append(cache.get_or_convert(path, reader_fn))

        canonical_set, set_meshes, set_geometries, set_boundaries = (
            case_set_from_cfd_datasets(
                datasets,
                case_set_id=source_set.case_set_id,
                name=source_set.name,
                case_ids=[item.case_id for item in source_set.cases],
                case_names=[item.name for item in source_set.cases],
                parameters=[item.parameters for item in source_set.cases],
                parameter_units=source_set.parameter_units,
                geometry_ids=[item.geometry_id for item in source_set.cases],
                source_uris=paths,
            )
        )
        case_sets.append(canonical_set)
        _merge_unique(meshes, set_meshes, lambda item: item.mesh_id, "mesh")
        _merge_unique(
            geometries,
            set_geometries,
            lambda item: item.geometry_id,
            "geometry",
        )
        _merge_unique(
            boundaries,
            set_boundaries,
            lambda item: item.patch_id,
            "boundary",
        )

    return TwinProject(
        project_id=source.project_id,
        name=source.name,
        case_sets=tuple(case_sets),
        meshes=tuple(meshes.values()),
        geometries=tuple(geometries.values()),
        boundaries=tuple(boundaries.values()),
        schema_version=SCHEMA_VERSION,
        metadata={**dict(source.metadata), "source_manifest_schema": source.schema},
    )


def ingest_project_source_manifest(
    path: str | Path,
    **kwargs: Any,
) -> TwinProject:
    """Load a source manifest and resolve relative paths beside it."""

    manifest_path = Path(path).expanduser().resolve()
    source = load_project_source_manifest(manifest_path)
    return ingest_project_source(source, base_dir=manifest_path.parent, **kwargs)


__all__ = [
    "SOURCE_MANIFEST_SCHEMA",
    "CaseSetSource",
    "CaseSource",
    "ProjectSource",
    "ingest_project_source",
    "ingest_project_source_manifest",
    "load_project_source_manifest",
    "project_source_from_dict",
    "save_project_source_manifest",
]
