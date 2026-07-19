"""Tests for mesh-aware scalar, vector, and temporal field validation."""

from __future__ import annotations

import numpy as np
import pytest

from naviertwin.core.validation.comparison import (
    ComparisonContext,
    ComparisonStatus,
    compare_fields,
    compare_temporal_rollout,
)


def test_scalar_comparison_applies_mask_and_conservation_weights() -> None:
    truth = np.array([1.0, 2.0, np.nan, 4.0])
    prediction = np.array([1.0, 3.0, 99.0, 5.0])
    result = compare_fields(
        truth,
        prediction,
        valid_mask=[True, True, True, False],
        weights=[1.0, 2.0, 3.0, 4.0],
    )

    assert result.comparable
    np.testing.assert_array_equal(result.valid_mask, [True, True, False, False])
    np.testing.assert_allclose(result.absolute_error[:2], [0.0, 1.0])
    assert np.isnan(result.absolute_error[2:]).all()
    assert result.metrics["valid_fraction"] == pytest.approx(0.5)
    assert result.metrics["truth_integral"] == pytest.approx(5.0)
    assert result.metrics["prediction_integral"] == pytest.approx(7.0)
    assert result.metrics["integral_bias"] == pytest.approx(2.0)


def test_vector_comparison_reports_magnitude_and_angle() -> None:
    truth = np.array([[1.0, 0.0], [0.0, 1.0], [0.0, 0.0]])
    prediction = np.array([[0.0, 1.0], [0.0, 2.0], [1.0, 0.0]])
    result = compare_fields(truth, prediction, component_axis=-1)

    np.testing.assert_allclose(result.absolute_error, [np.sqrt(2.0), 1.0, 1.0])
    assert result.component_absolute_error.shape == truth.shape
    assert result.metrics["mean_angular_error_deg"] == pytest.approx(45.0)
    assert result.metrics["max_angular_error_deg"] == pytest.approx(90.0)


def test_different_meshes_require_explicit_mapping_provenance() -> None:
    context = ComparisonContext(truth_mesh_id="mesh-a", prediction_mesh_id="mesh-b")
    with pytest.raises(ValueError, match="mapping_id"):
        compare_fields([1.0], [1.0], context=context)

    mapped = ComparisonContext(
        truth_mesh_id="mesh-a",
        prediction_mesh_id="mesh-b",
        mapping_id="map-a-to-b-v1",
    )
    assert compare_fields([1.0], [1.0], context=mapped).comparable


def test_no_truth_returns_honest_non_comparable_result() -> None:
    result = compare_fields(
        None,
        np.ones(3),
        context=ComparisonContext(support_status="OUT_OF_SUPPORT"),
    )
    assert result.status is ComparisonStatus.EXTRAPOLATION
    assert not result.comparable
    assert result.absolute_error is None
    assert result.metrics == {}


def test_remap_floor_separates_resolvable_model_error() -> None:
    result = compare_fields(
        [1.0, 1.0],
        [1.1, 0.9],
        remap_relative_l2_floor=0.05,
    )
    total = result.metrics["relative_l2"]
    expected = np.sqrt(max(total**2 - 0.05**2, 0.0))
    assert result.metrics["model_relative_l2_above_remap"] == pytest.approx(expected)


def test_temporal_rollout_reports_step_errors_and_drift() -> None:
    truth = np.ones((2, 3))
    prediction = np.array([[1.0, 1.1, 1.2], [1.0, 1.1, 1.2]])
    result = compare_temporal_rollout(truth, prediction, times=[0.0, 1.0, 2.0])

    np.testing.assert_allclose(result.relative_l2_by_step, [0.0, 0.1, 0.2])
    assert result.metrics["final_relative_l2"] == pytest.approx(0.2)
    assert result.metrics["relative_l2_drift_per_time"] == pytest.approx(0.1)


def test_temporal_rollout_rejects_unordered_times() -> None:
    with pytest.raises(ValueError, match="strictly increasing"):
        compare_temporal_rollout(
            np.ones((2, 3)),
            np.ones((2, 3)),
            times=[0.0, 1.0, 0.5],
        )


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"valid_mask": [True]}, "valid_mask shape"),
        ({"weights": [1.0]}, "weights shape"),
        ({"remap_relative_l2_floor": -1.0}, "non-negative"),
    ],
)
def test_invalid_comparison_contracts_raise(kwargs: dict[str, object], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        compare_fields([1.0, 2.0], [1.0, 2.0], **kwargs)
