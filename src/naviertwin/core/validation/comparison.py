"""Mesh-aware field comparison contracts for digital-twin validation."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray

from naviertwin.core.validation.metrics import r2_score, relative_l2_error, rmse


class ComparisonStatus(str, Enum):
    """Availability state of a ground-truth comparison."""

    COMPARABLE = "comparable"
    EXTRAPOLATION = "extrapolation"
    NO_GROUND_TRUTH = "no_ground_truth"


@dataclass(frozen=True, slots=True)
class ComparisonContext:
    """Provenance needed to decide whether two fields are comparable."""

    truth_mesh_id: str = ""
    prediction_mesh_id: str = ""
    mapping_id: str = ""
    support_status: str = "UNKNOWN"
    metadata: dict[str, Any] = field(default_factory=dict)

    def validate_mesh_contract(self) -> None:
        """Require an explicit mapping when fields live on different meshes."""
        if (
            self.truth_mesh_id
            and self.prediction_mesh_id
            and self.truth_mesh_id != self.prediction_mesh_id
            and not self.mapping_id
        ):
            raise ValueError(
                "Fields on different meshes require an explicit mapping_id before "
                "validation."
            )


@dataclass(frozen=True, slots=True)
class FieldComparison:
    """A comparison result with masks, metrics, and provenance."""

    status: ComparisonStatus
    reason: str
    absolute_error: NDArray[np.float64] | None
    component_absolute_error: NDArray[np.float64] | None
    valid_mask: NDArray[np.bool_]
    metrics: dict[str, float]
    context: ComparisonContext

    @property
    def comparable(self) -> bool:
        """Return whether ground-truth metrics are available."""
        return self.status is ComparisonStatus.COMPARABLE


@dataclass(frozen=True, slots=True)
class TemporalComparison:
    """Per-step and aggregate metrics for an unsteady rollout."""

    times: NDArray[np.float64]
    relative_l2_by_step: NDArray[np.float64]
    rmse_by_step: NDArray[np.float64]
    metrics: dict[str, float]


def _normalise_component_axis(ndim: int, component_axis: int | None) -> int | None:
    if component_axis is None:
        return None
    axis = component_axis + ndim if component_axis < 0 else component_axis
    if axis < 0 or axis >= ndim:
        raise ValueError(f"component_axis {component_axis} is invalid for {ndim}D data.")
    return axis


def _sample_shape(shape: tuple[int, ...], component_axis: int | None) -> tuple[int, ...]:
    if component_axis is None:
        return shape
    return shape[:component_axis] + shape[component_axis + 1 :]


def _validated_mask(
    truth: NDArray[np.float64],
    prediction: NDArray[np.float64],
    valid_mask: ArrayLike | None,
    component_axis: int | None,
) -> NDArray[np.bool_]:
    sample_shape = _sample_shape(truth.shape, component_axis)
    if valid_mask is None:
        mask = np.ones(sample_shape, dtype=np.bool_)
    else:
        mask = np.asarray(valid_mask, dtype=np.bool_)
        if mask.shape != sample_shape:
            raise ValueError(
                f"valid_mask shape {mask.shape} does not match sample shape "
                f"{sample_shape}."
            )

    finite = np.isfinite(truth) & np.isfinite(prediction)
    if component_axis is not None:
        finite = np.all(finite, axis=component_axis)
    mask = mask & finite
    if not np.any(mask):
        raise ValueError("No finite, valid samples remain for field comparison.")
    return mask


def _selected_components(
    values: NDArray[np.float64],
    mask: NDArray[np.bool_],
    component_axis: int | None,
) -> NDArray[np.float64]:
    if component_axis is None:
        return values[mask].reshape(-1)
    component_last = np.moveaxis(values, component_axis, -1)
    return component_last[mask]


def _angular_error_degrees(
    truth: NDArray[np.float64], prediction: NDArray[np.float64]
) -> NDArray[np.float64]:
    truth_norm = np.linalg.norm(truth, axis=-1)
    prediction_norm = np.linalg.norm(prediction, axis=-1)
    nonzero = (truth_norm > 0.0) & (prediction_norm > 0.0)
    if not np.any(nonzero):
        return np.empty(0, dtype=np.float64)
    cosine = np.sum(truth[nonzero] * prediction[nonzero], axis=-1)
    cosine /= truth_norm[nonzero] * prediction_norm[nonzero]
    return np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0)))


def compare_fields(
    truth: ArrayLike | None,
    prediction: ArrayLike,
    *,
    valid_mask: ArrayLike | None = None,
    weights: ArrayLike | None = None,
    component_axis: int | None = None,
    context: ComparisonContext | None = None,
    remap_relative_l2_floor: float | None = None,
) -> FieldComparison:
    """Compare scalar or vector fields in a proven common comparison space.

    ``component_axis`` must be supplied for vector fields. Invalid/non-finite samples
    are excluded from metrics and represented as NaN in returned error arrays.
    Different mesh IDs are accepted only when ``context.mapping_id`` records the
    mapping used to establish a common space.
    """
    ctx = context or ComparisonContext()
    pred = np.asarray(prediction, dtype=np.float64)
    if pred.size == 0:
        raise ValueError("Prediction field is empty.")

    axis = _normalise_component_axis(pred.ndim, component_axis)
    sample_shape = _sample_shape(pred.shape, axis)
    if truth is None:
        mask = np.asarray(valid_mask, dtype=np.bool_) if valid_mask is not None else None
        if mask is None:
            mask = np.ones(sample_shape, dtype=np.bool_)
        elif mask.shape != sample_shape:
            raise ValueError(
                f"valid_mask shape {mask.shape} does not match sample shape "
                f"{sample_shape}."
            )
        status = (
            ComparisonStatus.EXTRAPOLATION
            if ctx.support_status == "OUT_OF_SUPPORT"
            else ComparisonStatus.NO_GROUND_TRUTH
        )
        return FieldComparison(
            status=status,
            reason="Ground truth is unavailable; error metrics were not computed.",
            absolute_error=None,
            component_absolute_error=None,
            valid_mask=mask,
            metrics={},
            context=ctx,
        )

    ctx.validate_mesh_contract()
    actual = np.asarray(truth, dtype=np.float64)
    if actual.size == 0:
        raise ValueError("Ground-truth field is empty.")
    if actual.shape != pred.shape:
        raise ValueError(
            f"Ground-truth shape {actual.shape} does not match prediction shape "
            f"{pred.shape}."
        )

    mask = _validated_mask(actual, pred, valid_mask, axis)
    selected_truth = _selected_components(actual, mask, axis)
    selected_prediction = _selected_components(pred, mask, axis)
    difference = selected_prediction - selected_truth

    metrics = {
        "rmse": float(rmse(selected_truth, selected_prediction)),
        "mae": float(np.mean(np.abs(difference))),
        "relative_l2": float(
            relative_l2_error(selected_truth, selected_prediction)
        ),
        "max_error": float(np.max(np.abs(difference))),
        "r2": float(r2_score(selected_truth, selected_prediction)),
        "valid_fraction": float(np.count_nonzero(mask) / mask.size),
    }

    component_error = np.full(actual.shape, np.nan, dtype=np.float64)
    if axis is None:
        component_error[mask] = np.abs(difference)
        absolute_error = component_error.copy()
    else:
        component_last_error = np.moveaxis(component_error, axis, -1)
        component_last_error[mask] = np.abs(difference)
        component_error = np.moveaxis(component_last_error, -1, axis)
        absolute_error = np.full(mask.shape, np.nan, dtype=np.float64)
        absolute_error[mask] = np.linalg.norm(difference, axis=-1)
        metrics["vector_rmse"] = float(
            np.sqrt(np.mean(np.sum(difference**2, axis=-1)))
        )
        angles = _angular_error_degrees(selected_truth, selected_prediction)
        if angles.size:
            metrics["mean_angular_error_deg"] = float(np.mean(angles))
            metrics["max_angular_error_deg"] = float(np.max(angles))

    if weights is not None:
        if axis is not None:
            raise ValueError("Conservation integrals currently require scalar fields.")
        weight_array = np.asarray(weights, dtype=np.float64)
        if weight_array.shape != mask.shape:
            raise ValueError(
                f"weights shape {weight_array.shape} does not match sample shape "
                f"{mask.shape}."
            )
        if np.any(weight_array[mask] < 0.0) or not np.all(
            np.isfinite(weight_array[mask])
        ):
            raise ValueError("weights must be finite and non-negative on valid samples.")
        truth_integral = float(np.sum(actual[mask] * weight_array[mask]))
        prediction_integral = float(np.sum(pred[mask] * weight_array[mask]))
        integral_bias = prediction_integral - truth_integral
        metrics.update(
            {
                "truth_integral": truth_integral,
                "prediction_integral": prediction_integral,
                "integral_bias": integral_bias,
                "integral_relative_error": float(
                    abs(integral_bias) / abs(truth_integral)
                    if truth_integral != 0.0
                    else abs(integral_bias)
                ),
            }
        )

    if remap_relative_l2_floor is not None:
        floor = float(remap_relative_l2_floor)
        if not np.isfinite(floor) or floor < 0.0:
            raise ValueError("remap_relative_l2_floor must be finite and non-negative.")
        total = metrics["relative_l2"]
        metrics["remap_relative_l2_floor"] = floor
        metrics["model_relative_l2_above_remap"] = float(
            np.sqrt(max(total * total - floor * floor, 0.0))
        )

    return FieldComparison(
        status=ComparisonStatus.COMPARABLE,
        reason="Ground truth and prediction share a validated comparison space.",
        absolute_error=absolute_error,
        component_absolute_error=component_error,
        valid_mask=mask,
        metrics=metrics,
        context=ctx,
    )


def compare_temporal_rollout(
    truth: ArrayLike,
    prediction: ArrayLike,
    *,
    times: ArrayLike | None = None,
    time_axis: int = -1,
) -> TemporalComparison:
    """Compute per-step drift metrics for an unsteady prediction rollout."""
    actual = np.asarray(truth, dtype=np.float64)
    pred = np.asarray(prediction, dtype=np.float64)
    if actual.size == 0 or pred.size == 0:
        raise ValueError("Temporal rollout fields must not be empty.")
    if actual.shape != pred.shape:
        raise ValueError(
            f"Ground-truth shape {actual.shape} does not match prediction shape "
            f"{pred.shape}."
        )
    axis = _normalise_component_axis(actual.ndim, time_axis)
    assert axis is not None
    n_steps = actual.shape[axis]
    time_values = (
        np.arange(n_steps, dtype=np.float64)
        if times is None
        else np.asarray(times, dtype=np.float64).reshape(-1)
    )
    if time_values.size != n_steps:
        raise ValueError(f"times has {time_values.size} values; expected {n_steps}.")
    if not np.all(np.isfinite(time_values)):
        raise ValueError("times must contain only finite values.")
    if n_steps > 1 and np.any(np.diff(time_values) <= 0.0):
        raise ValueError("times must be strictly increasing.")

    actual_steps = np.moveaxis(actual, axis, 0).reshape(n_steps, -1)
    pred_steps = np.moveaxis(pred, axis, 0).reshape(n_steps, -1)
    rel = np.asarray(
        [relative_l2_error(a, p) for a, p in zip(actual_steps, pred_steps)],
        dtype=np.float64,
    )
    step_rmse = np.asarray(
        [rmse(a, p) for a, p in zip(actual_steps, pred_steps)], dtype=np.float64
    )
    drift_slope = 0.0
    if n_steps > 1:
        drift_slope = float(np.polyfit(time_values, rel, deg=1)[0])
    return TemporalComparison(
        times=time_values,
        relative_l2_by_step=rel,
        rmse_by_step=step_rmse,
        metrics={
            "mean_relative_l2": float(np.mean(rel)),
            "final_relative_l2": float(rel[-1]),
            "peak_relative_l2": float(np.max(rel)),
            "mean_rmse": float(np.mean(step_rmse)),
            "relative_l2_drift_per_time": drift_slope,
        },
    )


__all__ = [
    "ComparisonContext",
    "ComparisonStatus",
    "FieldComparison",
    "TemporalComparison",
    "compare_fields",
    "compare_temporal_rollout",
]
