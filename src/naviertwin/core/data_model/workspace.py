"""Runtime owner for canonical projects and their in-memory CFD objects."""

from __future__ import annotations

import hashlib
import threading
import uuid
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
from numpy.typing import NDArray

from naviertwin.core.data_model.adapters import case_set_from_cfd_datasets
from naviertwin.core.data_model.manifest import save_project_manifest
from naviertwin.core.data_model.schema import LineageKind, LineageRecord, TwinProject


def _workspace_id(name: str, source: str) -> str:
    digest = hashlib.sha256(f"{name}\0{source}".encode("utf-8")).hexdigest()[:16]
    return f"workspace-{digest}"


@dataclass(frozen=True, slots=True)
class WorkspaceStatus:
    """Detached status used by GUI state and tests."""

    project_id: str
    project_name: str
    case_set_id: str
    case_count: int
    active_case_index: int
    parameter_names: tuple[str, ...]
    revision: int
    has_engine: bool
    runtime_complete: bool


class TwinWorkspace:
    """Single runtime owner of project, datasets, parameters, and trained engine.

    Canonical manifests describe durable identity and provenance. Reader-facing
    ``CFDDataset`` instances remain in this runtime layer because live VTK objects
    do not belong in a JSON schema.
    """

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._project: TwinProject | None = None
        self._datasets: tuple[Any, ...] = ()
        self._parameters: NDArray[np.float64] | None = None
        self._parameter_names: tuple[str, ...] = ()
        self._active_case_index = 0
        self._view_dataset: Any | None = None
        self._engine: Any | None = None
        self._runtime_complete = False
        self._revision = 0

    @property
    def project(self) -> TwinProject | None:
        return self._project

    @property
    def view_dataset(self) -> Any | None:
        return self._view_dataset

    @property
    def case_datasets(self) -> tuple[Any, ...] | None:
        if not self._runtime_complete or len(self._datasets) <= 1:
            return None
        return self._datasets

    @property
    def parameters(self) -> NDArray[np.float64] | None:
        return self._parameters

    @property
    def parameter_names(self) -> tuple[str, ...]:
        return self._parameter_names

    @property
    def engine(self) -> Any | None:
        return self._engine

    def load_single_dataset(
        self,
        dataset: Any,
        *,
        name: str = "single CFD case",
        source: str = "",
    ) -> TwinProject:
        """Replace workspace content with one steady or unsteady CFD case."""
        case_set, meshes, geometries, boundaries = case_set_from_cfd_datasets(
            [dataset],
            case_set_id="primary",
            name=name,
            case_ids=("case-0000",),
            case_names=(name,),
            source_uris=(source or None,),
        )
        project = TwinProject(
            project_id=_workspace_id(name, source),
            name=name,
            case_sets=(case_set,),
            meshes=meshes,
            geometries=geometries,
            boundaries=boundaries,
            metadata={"runtime_source": source},
        )
        with self._lock:
            self._project = project
            self._datasets = (dataset,)
            self._parameters = None
            self._parameter_names = ()
            self._active_case_index = 0
            self._view_dataset = dataset
            self._engine = None
            self._runtime_complete = True
            self._revision += 1
        return project

    def load_case_set(
        self,
        datasets: Sequence[Any],
        parameters: Any,
        parameter_names: Sequence[str],
        *,
        case_names: Sequence[str] | None = None,
        name: str = "CFD case set",
        source: str = "",
        geometry_ids: Sequence[str] | None = None,
        parameter_units: Mapping[str, str] | None = None,
    ) -> TwinProject:
        """Replace workspace content with a condition/geometry/time case set."""
        runtime_datasets = tuple(datasets)
        if not runtime_datasets:
            raise ValueError("case set must contain at least one runtime dataset")
        names = tuple(str(item) for item in parameter_names)
        rows = np.asarray(parameters, dtype=np.float64)
        if rows.ndim == 1:
            rows = rows.reshape(-1, 1)
        if rows.ndim != 2 or rows.shape != (len(runtime_datasets), len(names)):
            raise ValueError(
                "parameter matrix shape must be (number of cases, number of names)"
            )
        if not np.all(np.isfinite(rows)):
            raise ValueError("case parameters must be finite")
        display_names = (
            tuple(str(item) for item in case_names)
            if case_names is not None
            else tuple(f"case-{index:04d}" for index in range(len(runtime_datasets)))
        )
        parameter_rows = tuple(
            {key: float(value) for key, value in zip(names, row)} for row in rows
        )
        case_set, meshes, geometries, boundaries = case_set_from_cfd_datasets(
            runtime_datasets,
            case_set_id="primary",
            name=name,
            case_ids=tuple(f"case-{index:04d}" for index in range(len(runtime_datasets))),
            case_names=display_names,
            parameters=parameter_rows,
            parameter_units=parameter_units,
            geometry_ids=geometry_ids,
        )
        project = TwinProject(
            project_id=_workspace_id(name, source),
            name=name,
            case_sets=(case_set,),
            meshes=meshes,
            geometries=geometries,
            boundaries=boundaries,
            metadata={"runtime_source": source},
        )
        stored_rows = rows.copy()
        stored_rows.setflags(write=False)
        with self._lock:
            self._project = project
            self._datasets = runtime_datasets
            self._parameters = stored_rows
            self._parameter_names = names
            self._active_case_index = 0
            self._view_dataset = runtime_datasets[0]
            self._engine = None
            self._runtime_complete = True
            self._revision += 1
        return project

    def adopt_project(
        self,
        project: TwinProject,
        *,
        view_dataset: Any,
        engine: Any | None = None,
        runtime_complete: bool = False,
    ) -> None:
        """Attach a loaded manifest when only part of its arrays are materialized."""
        with self._lock:
            self._project = project
            self._datasets = (view_dataset,)
            self._parameters = None
            self._parameter_names = ()
            self._active_case_index = 0
            self._view_dataset = view_dataset
            self._engine = engine
            self._runtime_complete = runtime_complete
            self._revision += 1

    def select_case(self, index: int) -> Any:
        """Select a materialized case and return its dataset."""
        with self._lock:
            if not self._runtime_complete or not self._datasets:
                raise RuntimeError("workspace does not contain a complete runtime case set")
            if index < 0 or index >= len(self._datasets):
                raise IndexError(f"case index out of range: {index}")
            self._active_case_index = index
            self._view_dataset = self._datasets[index]
            self._revision += 1
            return self._view_dataset

    def set_view_dataset(self, dataset: Any) -> None:
        """Switch viewer output without replacing canonical training data."""
        with self._lock:
            self._view_dataset = dataset
            self._revision += 1

    def set_engine(self, engine: Any | None) -> None:
        with self._lock:
            self._engine = engine
            self._revision += 1

    def record_lineage(
        self,
        kind: LineageKind | str,
        *,
        artifact_id: str | None = None,
        input_ids: Sequence[str] = (),
        artifact_uri: str = "",
        checksum: str = "",
        strategy: str = "",
        parameters: Mapping[str, Any] | None = None,
        metrics: Mapping[str, float] | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> LineageRecord:
        """Append one immutable workflow artifact to the canonical project."""

        record = LineageRecord(
            artifact_id=artifact_id or uuid.uuid4().hex,
            kind=LineageKind(kind),
            input_ids=tuple(str(value) for value in input_ids),
            artifact_uri=str(artifact_uri),
            checksum=str(checksum),
            strategy=str(strategy),
            created_at=datetime.now(timezone.utc).isoformat(),
            parameters=dict(parameters or {}),
            metrics={str(key): float(value) for key, value in (metrics or {}).items()},
            metadata=dict(metadata or {}),
        )
        with self._lock:
            if self._project is None:
                raise RuntimeError("workspace has no canonical project")
            self._project = replace(
                self._project,
                lineage=(*self._project.lineage, record),
            )
            self._revision += 1
        return record

    def status(self) -> WorkspaceStatus:
        with self._lock:
            case_set_id = self._project.case_sets[0].case_set_id if self._project else ""
            return WorkspaceStatus(
                project_id=self._project.project_id if self._project else "",
                project_name=self._project.name if self._project else "",
                case_set_id=case_set_id,
                case_count=len(self._datasets),
                active_case_index=self._active_case_index,
                parameter_names=self._parameter_names,
                revision=self._revision,
                has_engine=self._engine is not None,
                runtime_complete=self._runtime_complete,
            )

    def save_manifest(self, path: str | Path) -> Path:
        with self._lock:
            if self._project is None:
                raise RuntimeError("workspace has no canonical project")
            project = self._project
        return save_project_manifest(project, path)


__all__ = ["TwinWorkspace", "WorkspaceStatus"]
