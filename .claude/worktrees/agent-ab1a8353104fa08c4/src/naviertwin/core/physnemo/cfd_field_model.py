"""CFD dataset based PhysicsNeMo-style field model.

The intended digital-twin training layout is:

``coordinates + operating parameters -> selected CFD output fields``.

Steady-state parameter sweeps can pass multiple CFD datasets plus a parameter
table where each row corresponds to one case. Time-series data maps each
timestep is treated as one case unless an explicit parameter table is supplied.
Vector fields keep their components (v5.0): U trains/predicts as U_x/U_y/U_z
channels so direction is preserved; magnitude is derived downstream for display.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True)
class CFDFieldTrainingData:
    """Prepared tensors used by supervised CFD field fitting.

    ``input_features`` carries **other CFD fields used as inputs** — unlike
    ``params`` (one row per snapshot), these vary per (snapshot, location), so
    they are stacked as ``(n_snapshots, n_locations, n_input_features)``. This
    turns the model into a field-to-field map, e.g. ``(x, U, t) -> p``.
    """

    coords: NDArray[np.float64]
    values: NDArray[np.float64]
    params: NDArray[np.float64]
    field_names: list[str]
    field_location: str
    source_components: list[int]
    output_fields: list[dict[str, object]]
    parameter_names: list[str]
    input_features: NDArray[np.float64] | None = None
    input_field_names: list[str] | None = None
    # 형상 가변(케이스마다 격자가 다름) — 격자에 진짜 구멍이 뚫린 CFD 결과는
    # 케이스마다 점 수가 다르므로 (n_snap, n_loc, n_out) 직사각 배열에 담기지
    # 않는다. 그때만 케이스별 블록을 싣고, coords/values 는 첫 케이스를 담아
    # predict(params) 같은 기존 경로를 유지한다.
    coords_blocks: list[NDArray[np.float64]] | None = None
    values_blocks: list[NDArray[np.float64]] | None = None

    @property
    def has_varying_mesh(self) -> bool:
        """케이스마다 격자가 다른가 (형상 가변)."""
        return self.coords_blocks is not None

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
        # 입력으로 쓰는 다른 CFD field (per-point) — 비면 좌표+파라미터만 입력.
        self.input_field_names: list[str] = []
        self.n_input_features: int = 0
        self.is_fitted: bool = False
        self.train_losses_: list[float] = []
        self.validation_metrics: dict[str, float] = {}
        self.training_metadata: dict[str, object] = {}

        self._coords: NDArray[np.float64] | None = None
        self._input_features: NDArray[np.float64] | None = None
        self._train_params: NDArray[np.float64] | None = None
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
        input_field_names: Sequence[str] | None = None,
        allow_varying_mesh: bool = False,
        hidden: int = 32,
        max_epochs: int = 150,
        max_train_points: int = 20_000,
    ) -> "PhysicsNeMoCFDFieldModel":
        """Build and fit a model from multiple CFD cases.

        Args:
            allow_varying_mesh: 케이스마다 격자가 달라도 학습한다 (형상 가변).
        """
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
            input_field_names=input_field_names,
            allow_varying_mesh=allow_varying_mesh,
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
        input_field_names: Sequence[str] | None = None,
        allow_varying_mesh: bool = False,
    ) -> None:
        """Train on coordinates, operating parameters, and selected fields.

        Args:
            input_field_names: other CFD fields fed as per-point inputs — makes
                this a field-to-field map, e.g. ``(x, U, t) -> p``. Predicting
                then requires those inputs (see :meth:`predict_at`); the stored
                training values are reused by :meth:`predict`.
        """
        import torch

        data = prepare_cfd_field_training_data_from_datasets(
            datasets,
            field_names=field_names,
            params=params,
            parameter_names=parameter_names,
            input_field_names=input_field_names,
            allow_varying_mesh=allow_varying_mesh,
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
        self.input_field_names = list(data.input_field_names or [])
        self.n_input_features = int(
            data.input_features.shape[2] if data.input_features is not None else 0
        )
        # 입력 field 시계열을 보관 → predict(params) 가 학습 파라미터 지점에서
        # 자기완결적으로 동작한다 (새 입력장은 predict_at 으로 명시 제공).
        self._input_features = (
            np.asarray(data.input_features, dtype=np.float64)
            if data.input_features is not None
            else None
        )
        self.input_dim = self.param_dim
        self.output_dim = int(data.coords.shape[0] * data.values.shape[2])
        self._param_min = np.min(data.params, axis=0)
        self._param_max = np.max(data.params, axis=0)
        self._train_params = np.asarray(data.params, dtype=np.float64)

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
        # 지표는 **채널 단위** (v5.0) — 벡터 U 는 U_x/U_y/U_z 각각. y 열 순서가
        # output_fields spec 순서와 같다는 것이 계약이다.
        per_field = {}
        for index, spec in enumerate(self.output_fields):
            per_field[str(spec["display_name"])] = _regression_metrics(
                y_val[:, index : index + 1],
                pred_val[:, index : index + 1],
            )
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
            "input_field_names": list(self.input_field_names),
            "n_input_features": self.n_input_features,
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
        """Predict full scalar output fields per operating parameter row.

        Evaluates on the training coordinates. Use :meth:`predict_at` to
        evaluate on a different mesh or resolution.

        입력 field 를 쓰는 모델이면 학습 때 보관해 둔 입력장을 파라미터 축으로
        보간해 자동으로 채운다 — 그래서 ③Twin 슬라이더가 수정 없이 동작한다.
        새로운 입력장으로 예측하려면 :meth:`predict_at` 에 명시로 넘길 것.
        """
        if self._coords is None:
            raise RuntimeError("PhysicsNeMoCFDFieldModel.fit_datasets() 먼저 호출")
        features = None
        if self.n_input_features:
            features = self._interpolate_input_features(params)
        return self.predict_at(self._coords, params, input_features=features)

    def _interpolate_input_features(
        self, params: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        """학습 입력장을 질의 파라미터에서 보간한다 (1D 파라미터일 때).

        파라미터가 여러 개(케이스 스윕)면 보간이 모호하므로 가장 가까운 학습
        지점의 입력장을 쓴다 — 근사임을 호출자가 알 수 있게 문서화한다.
        """
        assert self._input_features is not None
        assert self._train_params is not None
        query = np.asarray(params, dtype=np.float64)
        if query.ndim == 1:
            query = query.reshape(1, -1)
        train = self._train_params
        out = np.empty(
            (query.shape[0], self._input_features.shape[1], self.n_input_features),
            dtype=np.float64,
        )
        order = np.argsort(train[:, 0]) if train.shape[1] == 1 else None
        axis = train[order, 0] if order is not None else None
        sorted_feats = self._input_features[order] if order is not None else None

        for row in range(query.shape[0]):
            if axis is None:
                # 다차원 파라미터 — 보간이 모호해 최근접 학습 지점을 쓴다(근사).
                nearest = int(np.argmin(np.linalg.norm(train - query[row], axis=1)))
                out[row] = self._input_features[nearest]
                continue
            # 1D(시간 등) — 위치·특징 축을 통째로 선형 보간, 범위 밖은 끝값 클램프.
            hi = int(np.clip(np.searchsorted(axis, query[row, 0]), 1, len(axis) - 1))
            lo = hi - 1
            span = float(axis[hi] - axis[lo])
            weight = 0.0 if span == 0 else (float(query[row, 0]) - float(axis[lo])) / span
            weight = float(np.clip(weight, 0.0, 1.0))
            out[row] = (1.0 - weight) * sorted_feats[lo] + weight * sorted_feats[hi]
        return out

    def predict_at(
        self,
        coords: NDArray[np.float64],
        params: NDArray[np.float64],
        input_features: NDArray[np.float64] | None = None,
    ) -> NDArray[np.float64]:
        """Predict output fields at arbitrary coordinates.

        The model is a neural field over ``(x, y, z, [input fields], params)``,
        so it is not tied to the training mesh — it can be evaluated on any
        point cloud (a finer grid, a different mesh, probe points).

        Args:
            coords: ``(n_locations, 3)`` evaluation coordinates.
            params: operating parameters, ``(n_rows, param_dim)`` or
                ``(param_dim,)``.
            input_features: required when the model was trained with
                ``input_field_names`` — the input fields sampled at ``coords``,
                shaped ``(n_rows, n_locations, n_input_features)`` or
                ``(n_locations, n_input_features)`` for a single row.

        Returns:
            ``(n_locations * n_fields, n_rows)`` field-major stacked values,
            the same layout :meth:`predict` returns.

        Raises:
            RuntimeError: model is not fitted.
            ValueError: coordinate/parameter/input-feature shapes do not match.
        """
        if not self.is_fitted or self._model is None:
            raise RuntimeError("PhysicsNeMoCFDFieldModel.fit_datasets() 먼저 호출")
        params_arr = np.asarray(params, dtype=np.float64)
        if params_arr.ndim == 1:
            params_arr = params_arr.reshape(1, -1)
        if params_arr.shape[1] != self.param_dim:
            raise ValueError(
                f"parameter dimension mismatch: expected {self.param_dim}, "
                f"got {params_arr.shape[1]}"
            )
        coords_arr = np.asarray(coords, dtype=np.float64)
        if coords_arr.ndim != 2 or coords_arr.shape[1] != self.coord_dim:
            raise ValueError(
                f"coords must be (n_locations, {self.coord_dim}), "
                f"got shape {coords_arr.shape}"
            )

        n_rows = params_arr.shape[0]
        n_locations = coords_arr.shape[0]
        tiled = np.tile(coords_arr, (n_rows, 1))
        repeated = np.repeat(params_arr, n_locations, axis=0)

        blocks = [tiled]
        if self.n_input_features:
            if input_features is None:
                raise ValueError(
                    "이 모델은 입력 field "
                    f"({', '.join(self.input_field_names)}) 가 필요합니다 — "
                    "predict_at(..., input_features=...) 로 넘기세요."
                )
            feats = np.asarray(input_features, dtype=np.float64)
            if feats.ndim == 2:
                feats = feats[np.newaxis, ...]
            expected = (n_rows, n_locations, self.n_input_features)
            if feats.shape != expected:
                raise ValueError(
                    f"input_features shape mismatch: expected {expected}, "
                    f"got {feats.shape}"
                )
            blocks.append(feats.reshape(n_rows * n_locations, self.n_input_features))
        elif input_features is not None:
            raise ValueError("이 모델은 입력 field 를 쓰지 않습니다.")
        blocks.append(repeated)
        # 열 순서는 _build_training_matrix 와 반드시 같아야 한다.
        X = np.hstack(blocks).astype(np.float32, copy=False)
        assert self._x_mean is not None
        assert self._x_std is not None
        assert self._y_mean is not None
        assert self._y_std is not None
        X_n = (X - self._x_mean) / self._x_std
        y = self._predict_matrix(X_n) * self._y_std + self._y_mean
        # 출력 열 수 = **채널** 수 (벡터 성분 포함, v5.0) — field 수가 아니다.
        n_channels = max(1, len(self.output_fields))
        y_blocks = y.reshape(n_rows, n_locations, n_channels)
        return y_blocks.transpose(0, 2, 1).reshape(n_rows, -1).T

    def _build_training_matrix(
        self,
        data: CFDFieldTrainingData,
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """X = [coords | input fields | params], Y = output fields.

        입력 field 는 (스냅샷, 위치)마다 값이 다르므로 params 처럼 repeat 하지
        않고 그대로 펼친다 — 열 순서는 :meth:`predict_at` 과 반드시 일치해야 한다.
        """
        if data.has_varying_mesh:
            # 형상 가변: 케이스마다 점 수가 달라 tile 이 안 된다. 신경장은 결국
            # (좌표, 파라미터) → 필드 의 행 집합이므로 케이스별로 펼쳐 쌓으면 된다.
            assert data.coords_blocks is not None and data.values_blocks is not None
            coord_rows: list[NDArray[np.float64]] = []
            value_rows: list[NDArray[np.float64]] = []
            repeats: list[int] = []
            for coords_i, values_i in zip(data.coords_blocks, data.values_blocks):
                n_snap_i, n_loc_i, n_out = values_i.shape
                coord_rows.append(np.tile(coords_i, (n_snap_i, 1)))
                value_rows.append(values_i.reshape(n_snap_i * n_loc_i, n_out))
                repeats.extend([n_loc_i] * n_snap_i)
            coords = np.vstack(coord_rows)
            values = np.vstack(value_rows)
            params = np.repeat(data.params, repeats, axis=0)
            return np.hstack([coords, params]), values

        n_steps, n_locations, _n_outputs = data.values.shape
        coords = np.tile(data.coords, (n_steps, 1))
        params = np.repeat(data.params, n_locations, axis=0)
        values = data.values.reshape(n_steps * n_locations, _n_outputs)
        blocks = [coords]
        if data.input_features is not None:
            n_in = data.input_features.shape[2]
            blocks.append(data.input_features.reshape(n_steps * n_locations, n_in))
        blocks.append(params)
        return np.hstack(blocks), values

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
    input_field_names: Sequence[str] | None = None,
    allow_varying_mesh: bool = False,
) -> CFDFieldTrainingData:
    """Extract aligned coordinates, parameters, and output fields.

    Args:
        input_field_names: other CFD fields to feed as **inputs** (per-point),
            e.g. predict ``p`` from ``U``. Vector fields become magnitude, the
            same convention outputs use.
        allow_varying_mesh: 케이스마다 격자가 달라도 허용한다 (형상 가변). 신경장은
            (좌표, 파라미터) → 필드 라 격자에 묶이지 않으므로 원리적으로 가능하다.
            벽 자리에 셀이 아예 없는 실제 CFD 결과가 여기 해당한다. 이때는 케이스별
            블록으로 학습하며, per-point 입력 field 는 아직 지원하지 않는다.
    """
    dataset_list = list(datasets)
    if not dataset_list:
        raise ValueError("at least one CFD dataset is required")
    fields = list(filter(None, map(lambda field: str(field).strip(), field_names)))
    if not fields:
        raise ValueError("at least one output field is required")

    coords_ref: NDArray[np.float64] | None = None
    location_ref: str | None = None
    values_blocks: list[NDArray[np.float64]] = []
    coords_blocks: list[NDArray[np.float64]] = []
    components_ref: list[int] | None = None
    total_snapshots = 0
    varying_mesh = False

    dataset_index = 0
    while dataset_index < len(dataset_list):
        dataset = dataset_list[dataset_index]
        coords, location, values, components, _channels = _extract_dataset_outputs(
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
            if components != components_ref:
                raise ValueError("selected field component layout differs across cases")
            same_mesh = coords.shape == coords_ref.shape and np.allclose(coords, coords_ref)
            if not same_mesh:
                if not allow_varying_mesh:
                    raise ValueError(
                        "all CFD cases must share the same mesh coordinates/topology "
                        f"(case index {dataset_index}). 형상 가변 데이터를 학습하려면 "
                        "allow_varying_mesh=True 를 쓰세요 — 신경장은 좌표 기반이라 "
                        "격자가 달라도 됩니다."
                    )
                varying_mesh = True
        coords_blocks.append(coords)
        values_blocks.append(values)
        total_snapshots += values.shape[0]
        dataset_index += 1

    assert coords_ref is not None
    assert location_ref is not None
    assert components_ref is not None
    # 형상 가변이면 직사각 concat 이 불가능하다 — 케이스별 블록을 그대로 넘기고
    # values/coords 에는 첫 케이스만 담는다(predict(params) 의 기본 격자).
    values_all = values_blocks[0] if varying_mesh else np.concatenate(values_blocks, axis=0)

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

    # 입력으로 쓸 다른 CFD field 들 — 출력과 같은 방식으로 뽑되 (스냅샷, 위치)
    # 마다 값이 다르므로 params 가 아니라 per-point 특징으로 쌓는다.
    inputs = list(filter(None, map(lambda f: str(f).strip(), input_field_names or [])))
    input_all: NDArray[np.float64] | None = None
    if inputs and varying_mesh:
        raise ValueError(
            "형상 가변 케이스에는 per-point 입력 field 를 아직 쓸 수 없습니다 — "
            "입력 field 를 비우고 좌표+파라미터로 학습하세요."
        )
    if inputs:
        overlap = sorted(set(inputs) & set(fields))
        if overlap:
            raise ValueError(
                f"입력과 출력에 같은 field 를 쓸 수 없습니다: {', '.join(overlap)}"
            )
        input_blocks: list[NDArray[np.float64]] = []
        for dataset in dataset_list:
            coords_in, location_in, values_in, _components, _in_channels = (
                _extract_dataset_outputs(dataset, inputs)
            )
            if location_in != location_ref:
                raise ValueError(
                    "입력 field 의 위치가 출력과 다릅니다: "
                    f"{location_ref} vs {location_in}"
                )
            if coords_in.shape != coords_ref.shape:
                raise ValueError("입력 field 의 메쉬 크기가 출력과 다릅니다")
            input_blocks.append(values_in)
        input_all = np.concatenate(input_blocks, axis=0)
        if input_all.shape[0] != total_snapshots:
            raise ValueError(
                f"입력 field 스냅샷 수 불일치: {input_all.shape[0]} vs {total_snapshots}"
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
        input_features=input_all,
        input_field_names=inputs or None,
        coords_blocks=coords_blocks if varying_mesh else None,
        values_blocks=values_blocks if varying_mesh else None,
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
        _coords, _location, values, _components, _chs = _extract_dataset_outputs(
            dataset_list[dataset_index],
            field_names,
        )
        total += int(values.shape[0])
        dataset_index += 1
    return total


_COMPONENT_SUFFIXES = ("x", "y", "z")


def _component_channel_names(field_name: str, n_components: int) -> list[str]:
    """벡터 field 의 성분별 채널 이름 — U → U_x, U_y, U_z (4성분 이상은 c0…)."""
    if n_components <= 1:
        return [str(field_name)]
    if n_components <= len(_COMPONENT_SUFFIXES):
        return [f"{field_name}_{_COMPONENT_SUFFIXES[i]}" for i in range(n_components)]
    return [f"{field_name}_c{i}" for i in range(n_components)]


def _extract_dataset_outputs(
    dataset: Any,
    field_names: Sequence[str],
) -> tuple[NDArray[np.float64], str, NDArray[np.float64], list[int], list[str]]:
    """선택 field 들을 (coords, location, 채널 스택, per-field 성분 수, 채널명)으로.

    벡터 field 는 **성분별 채널로 보존**한다 (v5.0) — 예전에는 크기(norm)로
    뭉개서 U 가 방향을 잃었다. U(T, n, 3) → 채널 U_x/U_y/U_z 세 개.
    """
    location_ref: str | None = None
    coords_ref: NDArray[np.float64] | None = None
    channel_values: list[NDArray[np.float64]] = []
    channel_names: list[str] = []
    components: list[int] = []
    n_steps_ref: int | None = None

    for field_name in field_names:
        raw, location = _extract_raw_field(dataset, field_name)
        coords = _coords_for_location(dataset.mesh, location)
        values, source_components = _field_to_component_snapshots(raw)
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
        for k, name in enumerate(_component_channel_names(field_name, source_components)):
            channel_values.append(values[:, :, k])
            channel_names.append(name)
        components.append(source_components)

    assert coords_ref is not None
    assert location_ref is not None
    values_stack = np.stack(channel_values, axis=2)
    return coords_ref, location_ref, values_stack, components, channel_names


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


def _field_to_component_snapshots(
    raw: NDArray[np.float64],
) -> tuple[NDArray[np.float64], int]:
    """field 배열을 (n_steps, n_loc, n_comp) 로 정규화한다 — 성분 보존 (v5.0).

    예전 ``_field_to_scalar_snapshots`` 는 벡터를 norm 으로 뭉개 방향을 잃었다.
    이제 스칼라는 성분 1개, 벡터는 성분 C개로 그대로 나간다.
    """
    arr = np.asarray(raw, dtype=np.float64)
    if arr.ndim == 1:
        return arr.reshape(1, -1, 1), 1
    if arr.ndim == 2:
        return arr[:, :, None], 1
    if arr.ndim == 3:
        return arr, int(arr.shape[2])
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
    """출력 채널 spec — **채널(성분)당 1개** (v5.0, 방향 보존).

    벡터 U(3성분)는 U_x/U_y/U_z 세 spec 이 된다. ``field_name`` 은 원본 field
    이름을 유지하므로 소비자(:func:`~naviertwin.web.service.split_multi_prediction`)
    가 성분을 다시 묶어 크기(magnitude)를 파생할 수 있다.
    """
    if len(field_names) != len(components):
        raise ValueError("field_names and components must have the same length")
    specs: list[dict[str, object]] = []
    start = 0
    for field_name, n_components in zip(field_names, components):
        for k, channel in enumerate(
            _component_channel_names(str(field_name), int(n_components))
        ):
            end = start + n_locations
            specs.append(
                {
                    "field_name": str(field_name),
                    "display_name": channel,
                    "component_index": int(k),
                    "location": location,
                    "source_components": int(n_components),
                    "start": int(start),
                    "end": int(end),
                    "n_locations": int(n_locations),
                }
            )
            start = end
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
