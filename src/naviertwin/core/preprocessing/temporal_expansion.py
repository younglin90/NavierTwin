"""Expand unsteady CFD cases into steady samples with time as a parameter."""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np
from numpy.typing import NDArray


def _time_slice(raw: Any, index: int, n_steps: int) -> NDArray[np.float64] | None:
    values = np.asarray(raw)
    if values.ndim >= 2 and values.shape[0] == n_steps:
        return np.asarray(values[index], dtype=np.float64)
    if values.ndim >= 2 and values.shape[-1] == n_steps:
        return np.asarray(values[..., index], dtype=np.float64)
    return None


def expand_unsteady_case_snapshots(
    datasets: Sequence[Any],
    params: Any,
    param_names: Sequence[str],
    *,
    field_names: Sequence[str],
) -> tuple[list[Any], NDArray[np.float64], list[str], bool]:
    """Return one materialized dataset per ``(case, time)`` sample.

    Existing steady inputs are returned unchanged. Unsteady fields are read from
    ``metadata['time_series_fields']`` first, then scalar
    ``extract_field_snapshots`` as a compatibility fallback.
    """

    cases = list(datasets)
    rows = np.asarray(params, dtype=np.float64)
    if rows.ndim == 1:
        rows = rows.reshape(-1, 1)
    if rows.ndim != 2 or rows.shape[0] != len(cases):
        raise ValueError("params rows must match the number of CFD cases")
    names = [str(value) for value in param_names]
    if rows.shape[1] != len(names):
        raise ValueError("param_names length must match params columns")
    has_time = any(len(getattr(case, "time_steps", ()) or ()) > 1 for case in cases)
    if not has_time:
        return cases, rows, names, False

    from naviertwin.core.cfd_reader.base import CFDDataset

    expanded_cases: list[Any] = []
    expanded_rows: list[NDArray[np.float64]] = []
    requested_fields = [str(value) for value in field_names]
    for case_index, case in enumerate(cases):
        times = [float(value) for value in (getattr(case, "time_steps", ()) or [0.0])]
        metadata = dict(getattr(case, "metadata", {}) or {})
        series = dict(metadata.get("time_series_fields", {}) or {})
        fallback: dict[str, Any] = {}
        for name in requested_fields:
            if name not in series:
                try:
                    fallback[name] = case.extract_field_snapshots(name)
                except Exception:  # noqa: BLE001
                    pass
        for time_index, time_value in enumerate(times):
            mesh = case.mesh.copy(deep=True)
            for name in requested_fields:
                values = _time_slice(
                    series.get(name, fallback.get(name)), time_index, len(times)
                )
                if values is None:
                    continue
                if values.shape[0] == int(getattr(mesh, "n_points", -1)):
                    mesh.point_data[name] = values
                elif values.shape[0] == int(getattr(mesh, "n_cells", -1)):
                    mesh.cell_data[name] = values
                else:
                    raise ValueError(
                        f"time-series field {name!r} does not match mesh locations"
                    )
            expanded_cases.append(
                CFDDataset(
                    mesh=mesh,
                    time_steps=[time_value],
                    field_names=list(getattr(case, "field_names", requested_fields)),
                    metadata={
                        **metadata,
                        "source_case_index": case_index,
                        "source_time_index": time_index,
                        "source_time": time_value,
                    },
                )
            )
            expanded_rows.append(np.concatenate([rows[case_index], [time_value]]))
    return expanded_cases, np.stack(expanded_rows), [*names, "t"], True


__all__ = ["expand_unsteady_case_snapshots"]
