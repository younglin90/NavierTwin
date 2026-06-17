"""CFD dataset based PhysicsNeMo-style field model.

The intended digital-twin training layout is:

``coordinates + operating parameters -> selected CFD output fields``.

Steady-state parameter sweeps can pass multiple CFD datasets plus a parameter
table where each row corresponds to one case. Time-series data maps each
timestep is treated as one case unless an explicit parameter table is supplied.
Vector fields are converted to magnitude to support immediate contour display.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True)
class CFDFieldTrainingData:
    """Prepared tensors used by supervised CFD field fitting."""

    coords: NDArray[np.float64]
    values: NDArray[np.float64]
    params: NDArray[np.float64]
    field_names: list[str]
    field_location: str
    source_components: list[int]
    output_fields: list[dict[str, object]]
    parameter_names: list[str]

    @property
    def field_name(self) -> str:
        """Comma-separated output field label used by legacy callers."""
        return ",".join(self.field_names)

    @property
    def source_components_count(self) -> int:
        """Legacy single-field component count."""
        return self.source_components[0] if self.source_components else 1


class PhysicsNeMoCFDFieldModel:
    """Supervised CFD field model exposed through the Physics AI workflow."""

    def __init__(
        self,
        *,
        hidden: int = 32,
        n_layers: int = 3,
        max_epochs: int = 150,
        lr: float = 1e-3,
        max_train_points: int = 20_000,
        device: str = "auto",
        seed: int | None = 0,
    ) -> None:
        self.hidden = hidden
        self.n_layers = n_layers
        self.max_epochs = max_epochs
        self.lr = lr
        self.max_train_points = max_train_points
        self.device = device
        self.seed = seed

        self.input_dim: int = 1
        self.output_dim: int = 0
        self.param_dim: int = 1
        self.coord_dim: int = 3
        self.field_name: str = ""
        self.field_names: list[str] = []
        self.field_location: str = "point"
        self.source_components: int = 1
        self.source_components_by_field: list[int] = []
        self.output_fields: list[dict[str, object]] = []
        self.parameter_names: list[str] = []
        self.is_fitted: bool = False
        self.train_losses_: list[float] = []
        self.validation_metrics: dict[str, float] = {}
        self.training_metadata: dict[str, object] = {}

        self._coords: NDArray[np.float64] | None = None
        self._param_min: NDArray[np.float64] | None = None
        self._param_max: NDArray[np.float64] | None = None
        self._x_mean: NDArray[np.float32] | None = None
        self._x_std: NDArray[np.float32] | None = None
        self._y_mean: NDArray[np.float32] | None = None
        self._y_std: NDArray[np.float32] | None = None
        self._model: Any = None
        self._device: Any = None

    @classmethod
    def from_dataset(
        cls,
        dataset: Any,
        *,
        field_name: str,
        hidden: int = 32,
        max_epochs: int = 150,
        max_train_points: int = 20_000,
    ) -> "PhysicsNeMoCFDFieldModel":
        """Build and fit a model from one loaded CFD dataset."""
        return cls.from_datasets(
            [dataset],
            field_names=[field_name],
            hidden=hidden,
            max_epochs=max_epochs,
            max_train_points=max_train_points,
        )

    @classmethod
    def from_datasets(
        cls,
        datasets: Sequence[Any],
        *,
        field_names: Sequence[str],
        params: NDArray[np.float64] | None = None,
        parameter_names: Sequence[str] | None = None,
        hidden: int = 32,
        max_epochs: int = 150,
        max_train_points: int = 20_000,
    ) -> "PhysicsNeMoCFDFieldModel":
        """Build and fit a model from multiple CFD cases."""
        model = cls(
            hidden=hidden,
            max_epochs=max_epochs,
            max_train_points=max_train_points,
        )
        model.fit_datasets(
            datasets,
            field_names=field_names,
            params=params,
            parameter_names=parameter_names,
        )
        return model

    def fit_dataset(self, dataset: Any, *, field_name: str) -> None:
        """Train on one dataset and one selected CFD field."""
        self.fit_datasets([dataset], field_names=[field_name])

    def fit_datasets(
        self,
        datasets: Sequence[Any],
        *,
        field_names: Sequence[str],
        params: NDArray[np.float64] | None = None,
        parameter_names: Sequence[str] | None = None,
    ) -> None:
        """Train on coordinates, operating parameters, and selected fields."""
        import torch

        data = prepare_cfd_field_training_data_from_datasets(
            datasets,
            field_names=field_names,
            params=params,
            parameter_names=parameter_names,
        )
        X, y = self._build_training_matrix(data)
        train_idx, val_idx = self._sample_indices(X.shape[0])

        self._coords = data.coords.astype(np.float64, copy=True)
        self.field_name = data.field_name
        self.field_names = list(data.field_names)
        self.field_location = data.field_location
        self.source_components = data.source_components_count
        self.source_components_by_field = list(data.source_components)
        self.output_fields = list(data.output_fields)
        self.parameter_names = list(data.parameter_names)
        self.coord_dim = int(data.coords.shape[1])
        self.param_dim = int(data.params.shape[1])
        self.input_dim = self.param_dim
        self.output_dim = int(data.coords.shape[0] * data.values.shape[2])
        self._param_min = np.min(data.params, axis=0)
        self._param_max = np.max(data.params, axis=0)

        X_train = X[train_idx].astype(np.float32, copy=False)
        y_train = y[train_idx].astype(np.float32, copy=False)
        X_val = X[val_idx].astype(np.float32, copy=False) if val_idx.size else X_train
        y_val = y[val_idx].astype(np.float32, copy=False) if val_idx.size else y_train

        self._x_mean = X_train.mean(axis=0, keepdims=True)
        self._x_std = X_train.std(axis=0, keepdims=True)
        self._x_std[self._x_std < 1e-8] = 1.0
        self._y_mean = y_train.mean(axis=0, keepdims=True)
        self._y_std = y_train.std(axis=0, keepdims=True)
        self._y_std[self._y_std < 1e-8] = 1.0

        X_train_n = (X_train - self._x_mean) / self._x_std
        y_train_n = (y_train - self._y_mean) / self._y_std
        X_val_n = (X_val - self._x_mean) / self._x_std

        if self.seed is not None:
            torch.manual_seed(self.seed)
        self._device = self._resolve_device()
        self._model = self._build_network(X_train.shape[1], y_train.shape[1]).to(
            self._device
        )
        optim = torch.optim.Adam(self._model.parameters(), lr=self.lr)
        mse = torch.nn.MSELoss()
        tx = torch.tensor(X_train_n, dtype=torch.float32, device=self._device)
        ty = torch.tensor(y_train_n, dtype=torch.float32, device=self._device)

        self.train_losses_ = []
        epoch = 0
        while epoch < self.max_epochs:
            optim.zero_grad()
            pred = self._model(tx)
            loss = mse(pred, ty)
            loss.backward()
            optim.step()
            self.train_losses_.append(float(loss.item()))
            epoch += 1

        pred_val = self._predict_matrix(X_val_n)
        pred_val = pred_val * self._y_std + self._y_mean
        self.validation_metrics = _regression_metrics(y_val, pred_val)
        per_field = {}
        index = 0
        while index < len(self.field_names):
            field = self.field_names[index]
            per_field[field] = _regression_metrics(
                y_val[:, index : index + 1],
                pred_val[:, index : index + 1],
            )
            index += 1
        self.is_fitted = True
        self.training_metadata = {
            "source": "cfd_dataset",
            "physics_model": "PhysicsNeMo CFD Field",
            "backend": "torch_supervised",
            "physicsnemo_available": _physicsnemo_available(),
            "field_name": self.field_name,
            "field_names": list(self.field_names),
            "field_location": self.field_location,
            "source_components": list(self.source_components_by_field),
            "output_fields": list(self.output_fields),
            "n_params": self.param_dim,
            "n_outputs": self.output_dim,
            "n_field_outputs": len(self.field_names),
            "n_locations": int(data.coords.shape[0]),
            "n_snapshots": int(data.params.shape[0]),
            "n_train_rows": int(train_idx.size),
            "n_validation_rows": int(val_idx.size),
            "parameter_names": list(self.parameter_names),
            "parameter_min": self._param_min.astype(float).tolist(),
            "parameter_max": self._param_max.astype(float).tolist(),
            "validation_metrics": dict(self.validation_metrics),
            "per_field_metrics": per_field,
        }

    def predict(self, params: NDArray[np.float64]) -> NDArray[np.float64]:
        """Predict full scalar output fields per operating parameter row."""
        if not self.is_fitted or self._model is None or self._coords is None:
            raise RuntimeError("PhysicsNeMoCFDFieldModel.fit_datasets() 먼저 호출")
        params_arr = np.asarray(params, dtype=np.float64)
        if params_arr.ndim == 1:
            params_arr = params_arr.reshape(1, -1)
        if params_arr.shape[1] != self.param_dim:
            raise ValueError(
                f"parameter dimension mismatch: expected {self.param_dim}, "
                f"got {params_arr.shape[1]}"
            )

        n_rows = params_arr.shape[0]
        n_locations = self._coords.shape[0]
        coords = np.tile(self._coords, (n_rows, 1))
        repeated = np.repeat(params_arr, n_locations, axis=0)
        X = np.hstack([coords, repeated]).astype(np.float32, copy=False)
        assert self._x_mean is not None
        assert self._x_std is not None
        assert self._y_mean is not None
        assert self._y_std is not None
        X_n = (X - self._x_mean) / self._x_std
        y = self._predict_matrix(X_n) * self._y_std + self._y_mean
        y_blocks = y.reshape(n_rows, n_locations, len(self.field_names))
        return y_blocks.transpose(0, 2, 1).reshape(n_rows, -1).T

    def _build_training_matrix(
        self,
        data: CFDFieldTrainingData,
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        n_steps, n_locations, _n_outputs = data.values.shape
        coords = np.tile(data.coords, (n_steps, 1))
        params = np.repeat(data.params, n_locations, axis=0)
        values = data.values.reshape(n_steps * n_locations, _n_outputs)
        return np.hstack([coords, params]), values

    def _sample_indices(
        self,
        n_rows: int,
    ) -> tuple[NDArray[np.int64], NDArray[np.int64]]:
        rng = np.random.default_rng(self.seed)
        indices = np.arange(n_rows, dtype=np.int64)
        if n_rows > self.max_train_points:
            indices = rng.choice(indices, size=self.max_train_points, replace=False)
            indices = np.sort(indices)
        if indices.size < 5:
            return indices, np.array([], dtype=np.int64)
        rng.shuffle(indices)
        val_count = max(1, int(round(indices.size * 0.2)))
        return indices[val_count:], indices[:val_count]

    def _resolve_device(self) -> Any:
        import torch

        if self.device == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return torch.device(self.device)

    def _build_network(self, input_dim: int, output_dim: int) -> Any:
        import torch.nn as nn

        layers: list[nn.Module] = [nn.Linear(input_dim, self.hidden), nn.Tanh()]
        layer_idx = 0
        while layer_idx < max(0, self.n_layers - 1):
            layers.extend([nn.Linear(self.hidden, self.hidden), nn.Tanh()])
            layer_idx += 1
        layers.append(nn.Linear(self.hidden, output_dim))
        return nn.Sequential(*layers)

    def _predict_matrix(self, X: NDArray[np.float32]) -> NDArray[np.float32]:
        import torch

        assert self._model is not None
        assert self._device is not None
        with torch.no_grad():
            pred = self._model(torch.tensor(X, dtype=torch.float32, device=self._device))
        return pred.cpu().numpy().astype(np.float32, copy=False)


def prepare_cfd_field_training_data(
    dataset: Any,
    *,
    field_name: str,
) -> CFDFieldTrainingData:
    """Extract one field from a CFDDataset used by legacy callers."""
    return prepare_cfd_field_training_data_from_datasets(
        [dataset],
        field_names=[field_name],
    )


def prepare_cfd_field_training_data_from_datasets(
    datasets: Sequence[Any],
    *,
    field_names: Sequence[str],
    params: NDArray[np.float64] | None = None,
    parameter_names: Sequence[str] | None = None,
) -> CFDFieldTrainingData:
    """Extract aligned coordinates, parameters, and output fields."""
    dataset_list = list(datasets)
    if not dataset_list:
        raise ValueError("at least one CFD dataset is required")
    fields = list(filter(None, map(lambda field: str(field).strip(), field_names)))
    if not fields:
        raise ValueError("at least one output field is required")

    coords_ref: NDArray[np.float64] | None = None
    location_ref: str | None = None
    values_blocks: list[NDArray[np.float64]] = []
    components_ref: list[int] | None = None
    total_snapshots = 0

    dataset_index = 0
    while dataset_index < len(dataset_list):
        dataset = dataset_list[dataset_index]
        coords, location, values, components = _extract_dataset_outputs(
            dataset,
            fields,
        )
        if coords_ref is None:
            coords_ref = coords
            location_ref = location
            components_ref = components
        else:
            if location != location_ref:
                raise ValueError(
                    f"mixed output locations are not supported: {location_ref}, {location}"
                )
            if coords.shape != coords_ref.shape or not np.allclose(coords, coords_ref):
                raise ValueError(
                    "all CFD cases must share the same mesh coordinates/topology "
                    f"(case index {dataset_index})"
                )
            if components != components_ref:
                raise ValueError("selected field component layout differs across cases")
        values_blocks.append(values)
        total_snapshots += values.shape[0]
        dataset_index += 1

    assert coords_ref is not None
    assert location_ref is not None
    assert components_ref is not None
    values_all = np.concatenate(values_blocks, axis=0)

    if params is None:
        params_arr = _default_params_for_datasets(dataset_list, total_snapshots)
        param_names = ["time_or_case"]
    else:
        params_arr = np.asarray(params, dtype=np.float64)
        if params_arr.ndim == 1:
            params_arr = params_arr.reshape(-1, 1)
        if params_arr.ndim != 2:
            raise ValueError(f"params must be 2D, got shape {params_arr.shape}")
        if params_arr.shape[0] != total_snapshots:
            raise ValueError(
                f"params row count mismatch: {params_arr.shape[0]} vs "
                f"CFD snapshots {total_snapshots}"
        )
        param_names = (
            list(map(str, parameter_names))
            if parameter_names is not None
            else list(map(lambda index: f"param_{index}", range(params_arr.shape[1])))
        )
    if len(param_names) != params_arr.shape[1]:
        raise ValueError("parameter_names length must match params dimension")

    output_fields = _build_output_field_specs(
        fields,
        location=location_ref,
        components=components_ref,
        n_locations=coords_ref.shape[0],
    )
    return CFDFieldTrainingData(
        coords=coords_ref,
        values=values_all,
        params=params_arr,
        field_names=fields,
        field_location=location_ref,
        source_components=components_ref,
        output_fields=output_fields,
        parameter_names=param_names,
    )


def load_parameter_table(
    path: str | Path,
    *,
    columns: Sequence[str] | None,
    expected_rows: int,
) -> tuple[NDArray[np.float64], list[str]]:
    """Load operating parameters from a CSV table."""
    try:
        import pandas as pd
    except ImportError as exc:
        raise RuntimeError("pandas required to read PhysicsNeMo parameter CSV") from exc

    df = pd.read_csv(path)
    if columns:
        names = list(filter(None, map(lambda column: str(column).strip(), columns)))
        missing = sorted(set(names) - set(map(str, df.columns)))
        if missing:
            raise ValueError(f"params CSV missing columns: {', '.join(missing)}")
    else:
        names = list(map(str, df.select_dtypes(include=["number"]).columns))
    if not names:
        raise ValueError("params CSV must contain at least one numeric input column")
    values = df[names].to_numpy(dtype=np.float64)
    if values.shape[0] != expected_rows:
        raise ValueError(
            f"params CSV row count mismatch: {values.shape[0]} vs CFD snapshots "
            f"{expected_rows}"
        )
    return values, names


def count_dataset_snapshots(datasets: Sequence[Any], field_names: Sequence[str]) -> int:
    """Count total training snapshots across selected fields."""
    total = 0
    dataset_list = list(datasets)
    dataset_index = 0
    while dataset_index < len(dataset_list):
        _coords, _location, values, _components = _extract_dataset_outputs(
            dataset_list[dataset_index],
            field_names,
        )
        total += int(values.shape[0])
        dataset_index += 1
    return total


def _extract_dataset_outputs(
    dataset: Any,
    field_names: Sequence[str],
) -> tuple[NDArray[np.float64], str, NDArray[np.float64], list[int]]:
    location_ref: str | None = None
    coords_ref: NDArray[np.float64] | None = None
    values_by_field: list[NDArray[np.float64]] = []
    components: list[int] = []
    n_steps_ref: int | None = None

    field_index = 0
    field_list = list(field_names)
    while field_index < len(field_list):
        field_name = field_list[field_index]
        raw, location = _extract_raw_field(dataset, field_name)
        coords = _coords_for_location(dataset.mesh, location)
        values, source_components = _field_to_scalar_snapshots(raw)
        if location_ref is None:
            location_ref = location
            coords_ref = coords
            n_steps_ref = values.shape[0]
        else:
            if location != location_ref:
                raise ValueError(
                    "all selected output fields must use the same location "
                    f"({location_ref} vs {location})"
                )
            if coords.shape != coords_ref.shape:
                raise ValueError("selected output fields use different mesh sizes")
            if n_steps_ref != values.shape[0]:
                raise ValueError("selected output fields have different snapshot counts")
        values_by_field.append(values)
        components.append(source_components)
        field_index += 1

    assert coords_ref is not None
    assert location_ref is not None
    values_stack = np.stack(values_by_field, axis=2)
    return coords_ref, location_ref, values_stack, components


def _extract_raw_field(dataset: Any, field_name: str) -> tuple[NDArray[np.float64], str]:
    mesh = dataset.mesh
    metadata = getattr(dataset, "metadata", {}) or {}
    time_series = metadata.get("time_series_fields")
    locations = metadata.get("time_series_locations")

    if isinstance(time_series, dict) and field_name in time_series:
        raw = np.asarray(time_series[field_name], dtype=np.float64)
        location = (
            str(locations.get(field_name, "point"))
            if isinstance(locations, dict)
            else "point"
        )
    elif field_name in mesh.point_data:
        raw = np.asarray(mesh.point_data[field_name], dtype=np.float64)[None, ...]
        location = "point"
    elif field_name in mesh.cell_data:
        raw = np.asarray(mesh.cell_data[field_name], dtype=np.float64)[None, ...]
        location = "cell"
    else:
        raise ValueError(f"field '{field_name}' is not available in the CFD dataset")
    return raw, location


def _coords_for_location(mesh: Any, location: str) -> NDArray[np.float64]:
    if location == "cell":
        centers = mesh.cell_centers()
        coords = np.asarray(centers.points, dtype=np.float64)
    else:
        coords = np.asarray(mesh.points, dtype=np.float64)
    if coords.ndim != 2:
        raise ValueError(f"mesh coordinates must be 2D, got shape {coords.shape}")
    if coords.shape[1] < 3:
        pad = np.zeros((coords.shape[0], 3 - coords.shape[1]), dtype=np.float64)
        coords = np.hstack([coords, pad])
    return coords[:, :3]


def _field_to_scalar_snapshots(raw: NDArray[np.float64]) -> tuple[NDArray[np.float64], int]:
    arr = np.asarray(raw, dtype=np.float64)
    if arr.ndim == 1:
        return arr.reshape(1, -1), 1
    if arr.ndim == 2:
        return arr, 1
    if arr.ndim == 3:
        return np.linalg.norm(arr, axis=2), int(arr.shape[2])
    raise ValueError(f"unsupported field array shape: {arr.shape}")


def _default_params_for_datasets(
    datasets: Sequence[Any],
    total_snapshots: int,
) -> NDArray[np.float64]:
    if len(datasets) == 1:
        return _params_for_dataset(datasets[0], total_snapshots)
    if total_snapshots == 1:
        values = np.array([0.0], dtype=np.float64)
    else:
        values = np.linspace(0.0, 1.0, total_snapshots, dtype=np.float64)
    return values.reshape(-1, 1)


def _params_for_dataset(dataset: Any, n_steps: int) -> NDArray[np.float64]:
    raw_time_steps = getattr(dataset, "time_steps", [])
    if raw_time_steps is None:
        raw_time_steps = []
    time_steps = np.asarray(raw_time_steps, dtype=np.float64)
    if time_steps.size == n_steps:
        values = time_steps
    elif n_steps == 1:
        values = np.array([0.0], dtype=np.float64)
    else:
        values = np.linspace(0.0, 1.0, n_steps, dtype=np.float64)
    return values.reshape(-1, 1)


def _build_output_field_specs(
    field_names: Sequence[str],
    *,
    location: str,
    components: Sequence[int],
    n_locations: int,
) -> list[dict[str, object]]:
    if len(field_names) != len(components):
        raise ValueError("field_names and components must have the same length")
    specs: list[dict[str, object]] = []
    start = 0
    spec_index = 0
    while spec_index < len(field_names):
        field_name = field_names[spec_index]
        n_components = components[spec_index]
        display_name = f"{field_name}_mag" if int(n_components) > 1 else str(field_name)
        end = start + n_locations
        specs.append(
            {
                "field_name": str(field_name),
                "display_name": display_name,
                "location": location,
                "source_components": int(n_components),
                "start": int(start),
                "end": int(end),
                "n_locations": int(n_locations),
            }
        )
        start = end
        spec_index += 1
    return specs


def _flatten_field_major(values: NDArray[np.float32]) -> NDArray[np.float32]:
    arr = np.asarray(values, dtype=np.float32)
    if arr.ndim != 2:
        raise ValueError(f"prediction values must be 2D, got shape {arr.shape}")
    return arr.T.reshape(-1)


def _regression_metrics(
    truth: NDArray[np.float32],
    pred: NDArray[np.float32],
) -> dict[str, float]:
    truth64 = np.asarray(truth, dtype=np.float64)
    pred64 = np.asarray(pred, dtype=np.float64)
    err = truth64 - pred64
    rmse = float(np.sqrt(np.mean(err**2)))
    max_error = float(np.max(np.abs(err))) if err.size else 0.0
    denom = float(np.sum((truth64 - np.mean(truth64)) ** 2))
    r2 = 1.0 - float(np.sum(err**2)) / denom if denom > 0 else 1.0
    return {"rmse": rmse, "max_error": max_error, "r2": float(r2)}


def _physicsnemo_available() -> bool:
    try:
        import physicsnemo  # noqa: F401
    except ImportError:
        return False
    return True


__all__ = [
    "CFDFieldTrainingData",
    "PhysicsNeMoCFDFieldModel",
    "count_dataset_snapshots",
    "load_parameter_table",
    "prepare_cfd_field_training_data",
    "prepare_cfd_field_training_data_from_datasets",
]
