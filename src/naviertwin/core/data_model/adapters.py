"""Compatibility adapters from reader-facing ``CFDDataset`` objects."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from naviertwin.core.data_model.schema import (
    BoundaryCondition,
    BoundaryKind,
    BoundaryPatch,
    Case,
    CaseSet,
    FieldDataRef,
    FieldDescriptor,
    FieldLocation,
    GeometryDescriptor,
    MeshDescriptor,
    Snapshot,
)
from naviertwin.core.data_model.signature import compute_signature


def _slug(value: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9_.-]+", "-", value.strip()).strip("-")
    return slug or "item"


def _source_uri(dataset: Any, explicit: str | Path | None) -> str:
    if explicit is not None:
        return str(explicit)
    metadata = getattr(dataset, "metadata", {}) or {}
    for key in ("source_uri", "source_file", "case_dir", "foam_file"):
        if metadata.get(key):
            return str(metadata[key])
    return "memory://cfddataset"


def _dimensions(dataset: Any) -> tuple[int, int]:
    metadata = getattr(dataset, "metadata", {}) or {}
    explicit = metadata.get("topological_dim")
    if explicit is not None:
        topological = int(explicit)
    else:
        bounds = dataset.mesh.bounds
        spans = [abs(bounds[2 * idx + 1] - bounds[2 * idx]) for idx in range(3)]
        scale = max(max(spans), 1e-12)
        topological = max(1, sum(span > scale * 1e-9 for span in spans))
    points = np.asarray(dataset.mesh.points)
    embedding = int(points.shape[1]) if points.ndim == 2 else 3
    return topological, min(3, max(1, embedding))


def _field_array(dataset: Any, name: str) -> tuple[np.ndarray, FieldLocation]:
    time_series = (getattr(dataset, "metadata", {}) or {}).get(
        "time_series_fields", {}
    )
    locations = (getattr(dataset, "metadata", {}) or {}).get(
        "time_series_locations", {}
    )
    if isinstance(time_series, dict) and name in time_series:
        raw_location = locations.get(name, "point") if isinstance(locations, dict) else "point"
        return np.asarray(time_series[name]), FieldLocation(raw_location)
    if name in dataset.mesh.point_data:
        return np.asarray(dataset.mesh.point_data[name]), FieldLocation.POINT
    if name in dataset.mesh.cell_data:
        return np.asarray(dataset.mesh.cell_data[name]), FieldLocation.CELL
    raise ValueError(f"field {name!r} is not present in the CFD dataset")


def _component_names(n_components: int) -> tuple[str, ...]:
    if n_components == 1:
        return ()
    if n_components <= 3:
        return ("x", "y", "z")[:n_components]
    return tuple(f"c{index}" for index in range(n_components))


def _field_descriptors(dataset: Any) -> tuple[FieldDescriptor, ...]:
    metadata = getattr(dataset, "metadata", {}) or {}
    units = metadata.get("field_units", {})
    time_series = metadata.get("time_series_fields", {})
    descriptors: list[FieldDescriptor] = []
    for name in getattr(dataset, "field_names", []):
        array, location = _field_array(dataset, str(name))
        n_components = 1
        is_time_series = isinstance(time_series, dict) and name in time_series
        if is_time_series and array.ndim >= 3:
            n_components = int(array.shape[-1])
        elif not is_time_series and array.ndim >= 2:
            n_components = int(array.shape[-1])
        descriptors.append(
            FieldDescriptor(
                name=str(name),
                location=location,
                n_components=n_components,
                component_names=_component_names(n_components),
                unit=str(units.get(name, "")) if isinstance(units, dict) else "",
                dtype=str(array.dtype),
            )
        )
    return tuple(descriptors)


def _valid_mask_path(dataset: Any, location: FieldLocation) -> str | None:
    """Return a logical mask path only when the reader actually supplied one."""
    metadata = getattr(dataset, "metadata", {}) or {}
    if metadata.get("time_series_valid_mask") is not None:
        return f"masks/{location.value}/valid_mask"
    data = dataset.mesh.point_data if location is FieldLocation.POINT else dataset.mesh.cell_data
    if "valid_mask" in data or "vtkValidPointMask" in data:
        return f"masks/{location.value}/valid_mask"
    return None


def _boundary_kind(raw: Any, *, is_wall: bool = False) -> BoundaryKind:
    value = str(raw or "").lower()
    if is_wall or "wall" in value:
        return BoundaryKind.WALL
    for kind in BoundaryKind:
        if kind is not BoundaryKind.UNKNOWN and kind.value in value:
            return kind
    return BoundaryKind.UNKNOWN


def _boundaries(
    dataset: Any, mesh_id: str
) -> tuple[tuple[BoundaryPatch, ...], tuple[BoundaryCondition, ...]]:
    metadata = getattr(dataset, "metadata", {}) or {}
    raw_patches = metadata.get("boundary_patches", {})
    raw_conditions = metadata.get("boundary_conditions", {})
    if not isinstance(raw_patches, dict):
        raw_patches = {}
    if not isinstance(raw_conditions, dict):
        raw_conditions = {}
    patches: list[BoundaryPatch] = []
    conditions: list[BoundaryCondition] = []
    names = sorted(set(raw_patches) | set(raw_conditions))
    for name in names:
        info = raw_patches.get(name, {})
        if not isinstance(info, dict):
            info = {}
        condition = raw_conditions.get(name, {})
        if not isinstance(condition, dict):
            condition = {}
        kind = _boundary_kind(
            condition.get("type", info.get("type")),
            is_wall=bool(info.get("is_wall", False)),
        )
        patch_id = f"{mesh_id}:patch:{_slug(str(name))}"
        patches.append(
            BoundaryPatch(
                patch_id=patch_id,
                mesh_id=mesh_id,
                name=str(name),
                kind=kind,
                source="reader" if name in raw_patches else "user",
                metadata={
                    key: value
                    for key, value in info.items()
                    if key not in {"type", "is_wall"}
                },
            )
        )
        if condition:
            values = condition.get("values", {})
            if not isinstance(values, dict):
                values = {"value": values}
            units = condition.get("units", {})
            conditions.append(
                BoundaryCondition(
                    patch_id=patch_id,
                    kind=kind,
                    values=values,
                    units=units if isinstance(units, dict) else {},
                    source=str(condition.get("source", "user")),
                )
            )
    return tuple(patches), tuple(conditions)


def case_set_from_cfd_datasets(
    datasets: Sequence[Any],
    *,
    case_set_id: str,
    name: str,
    case_ids: Sequence[str] | None = None,
    case_names: Sequence[str] | None = None,
    parameters: Sequence[Mapping[str, float]] | None = None,
    parameter_units: Mapping[str, str] | None = None,
    geometry_ids: Sequence[str] | None = None,
    source_uris: Sequence[str | Path | None] | None = None,
) -> tuple[
    CaseSet,
    tuple[MeshDescriptor, ...],
    tuple[GeometryDescriptor, ...],
    tuple[BoundaryPatch, ...],
]:
    """Convert legacy datasets into a validated canonical case set graph."""

    items = list(datasets)
    if not items:
        raise ValueError("at least one CFD dataset is required")
    count = len(items)
    ids = list(case_ids) if case_ids is not None else [f"case-{idx:04d}" for idx in range(count)]
    names = list(case_names) if case_names is not None else list(ids)
    param_rows = list(parameters) if parameters is not None else [{} for _ in items]
    geometry_rows = list(geometry_ids) if geometry_ids is not None else ["" for _ in items]
    uri_rows = list(source_uris) if source_uris is not None else [None for _ in items]
    for label, values in (
        ("case_ids", ids),
        ("case_names", names),
        ("parameters", param_rows),
        ("geometry_ids", geometry_rows),
        ("source_uris", uri_rows),
    ):
        if len(values) != count:
            raise ValueError(f"{label} length must match datasets")

    fields = _field_descriptors(items[0])
    field_contract = {
        (item.name, item.location, item.n_components) for item in fields
    }
    meshes_by_id: dict[str, MeshDescriptor] = {}
    geometries_by_id: dict[str, GeometryDescriptor] = {}
    patches_by_id: dict[str, BoundaryPatch] = {}
    cases: list[Case] = []
    for index, dataset in enumerate(items):
        current_fields = _field_descriptors(dataset)
        current_contract = {
            (item.name, item.location, item.n_components) for item in current_fields
        }
        if current_contract != field_contract:
            raise ValueError("all cases must expose the same field contract")
        signature = compute_signature(dataset)
        mesh_id = f"mesh-{signature.topology_hash}-{signature.coordinate_hash}"
        inferred_geometry = f"geometry-mesh-{signature.coordinate_hash}"
        geometry_id = str(geometry_rows[index] or inferred_geometry)
        source_uri = _source_uri(dataset, uri_rows[index])
        dataset_metadata = getattr(dataset, "metadata", {}) or {}
        topological_dim, embedding_dim = _dimensions(dataset)
        meshes_by_id.setdefault(
            mesh_id,
            MeshDescriptor(
                mesh_id=mesh_id,
                geometry_id=geometry_id,
                topology_hash=signature.topology_hash,
                coordinate_hash=signature.coordinate_hash,
                topological_dim=topological_dim,
                embedding_dim=embedding_dim,
                n_points=signature.n_points,
                n_cells=signature.n_cells,
                source_uri=str(dataset_metadata.get("mesh_source_uri", "")),
                metadata={"geometry_id_inferred_from_mesh": not bool(geometry_rows[index])},
            ),
        )
        geometries_by_id.setdefault(
            geometry_id,
            GeometryDescriptor(
                geometry_id=geometry_id,
                source_uri=str(dataset_metadata.get("geometry_source_uri", "")),
                metadata={"identity_inferred_from_mesh": not bool(geometry_rows[index])},
            ),
        )
        patches, conditions = _boundaries(dataset, mesh_id)
        for patch in patches:
            patches_by_id.setdefault(patch.patch_id, patch)

        raw_time_values = getattr(dataset, "time_steps", None)
        raw_times = list(raw_time_values) if raw_time_values is not None else []
        if not raw_times:
            raw_times = [0.0]
        snapshots: list[Snapshot] = []
        for time_index, time_value in enumerate(raw_times):
            refs: list[FieldDataRef] = []
            for descriptor in fields:
                array, _ = _field_array(dataset, descriptor.name)
                refs.append(
                    FieldDataRef(
                        field_name=descriptor.name,
                        storage_uri=source_uri,
                        array_path=f"fields/{descriptor.location.value}/{descriptor.name}",
                        shape=tuple(int(size) for size in array.shape),
                        time_index=time_index if len(raw_times) > 1 else None,
                        valid_mask_path=_valid_mask_path(dataset, descriptor.location),
                    )
                )
            snapshots.append(
                Snapshot(
                    snapshot_id=f"{ids[index]}:t{time_index:06d}",
                    mesh_id=mesh_id,
                    time=float(time_value) if len(raw_times) > 1 else None,
                    fields=tuple(refs),
                )
            )
        cases.append(
            Case(
                case_id=str(ids[index]),
                name=str(names[index]),
                parameters={str(key): float(value) for key, value in param_rows[index].items()},
                snapshots=tuple(snapshots),
                boundary_conditions=conditions,
                metadata={"legacy_cfddataset": True},
            )
        )

    return (
        CaseSet(
            case_set_id=case_set_id,
            name=name,
            fields=fields,
            cases=tuple(cases),
            parameter_units=dict(parameter_units or {}),
        ),
        tuple(meshes_by_id.values()),
        tuple(geometries_by_id.values()),
        tuple(patches_by_id.values()),
    )


__all__ = ["case_set_from_cfd_datasets"]
