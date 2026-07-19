"""Reduction policy, memory estimate, and validity-mask tests."""

from __future__ import annotations

import numpy as np
import pytest

from naviertwin.core.preprocessing import (
    InterpolationMethod,
    PreprocessingPlan,
    ReductionMethod,
    ReductionTarget,
    apply_validity_mask,
    estimate_training_memory,
    resolution_for_retain_fraction,
    uniform_grid_dimensions,
)


def test_uniform_resolution_tracks_requested_ratio() -> None:
    bounds = (0.0, 2.0, 0.0, 1.0, 0.0, 0.0)
    resolution = resolution_for_retain_fraction(bounds, 20_000, 0.25)
    target_points = np.prod(uniform_grid_dimensions(bounds, resolution))

    assert abs(target_points - 5_000) / 5_000 < 0.05


def test_training_memory_includes_explicit_overhead() -> None:
    estimate = estimate_training_memory(
        points=1_000_000,
        snapshots=10,
        components=5,
        dtype_bytes=4,
        training_multiplier=6,
    )

    assert estimate.tensor_bytes == 200_000_000
    assert estimate.training_bytes == 1_200_000_000


def test_mask_is_preserved_and_invalid_vectors_are_filled() -> None:
    values = np.arange(12, dtype=np.float32).reshape(4, 3)
    filled, mask = apply_validity_mask(values, np.array([1, 0, 1, 0]))

    np.testing.assert_array_equal(mask, [True, False, True, False])
    np.testing.assert_array_equal(filled[1], np.zeros(3))
    np.testing.assert_array_equal(filled[3], np.zeros(3))
    np.testing.assert_array_equal(filled[0], values[0])


def test_reduction_target_requires_one_selector() -> None:
    with pytest.raises(ValueError, match="exactly one"):
        ReductionTarget(retain_fraction=0.5, max_points=100)


def test_wall_distance_plan_requires_boundary_preservation() -> None:
    with pytest.raises(ValueError, match="wall distance"):
        PreprocessingPlan(
            method=ReductionMethod.UNIFORM_GRID,
            target=ReductionTarget(retain_fraction=0.5),
            interpolation=InterpolationMethod.LINEAR,
            preserve_boundaries=False,
            generate_wall_distance=True,
        )


def test_coarsen_by_ratio_persists_valid_mask() -> None:
    pv = pytest.importorskip("pyvista")
    from naviertwin.core.cfd_reader.base import CFDDataset
    from naviertwin.web.service import coarsen_dataset

    mesh = pv.ImageData(dimensions=(31, 21, 1)).cast_to_unstructured_grid()
    mesh.point_data["p"] = np.linspace(0.0, 1.0, mesh.n_points)
    dataset = CFDDataset(mesh=mesh, time_steps=[0.0], field_names=["p"])

    result = coarsen_dataset(dataset, retain_fraction=0.25)
    reduced = result["dataset"]

    assert reduced.n_points < dataset.n_points
    assert reduced.n_points / dataset.n_points == pytest.approx(0.25, abs=0.03)
    assert "valid_mask" in reduced.mesh.point_data
    assert reduced.metadata["time_series_valid_mask"].shape == (1, reduced.n_points)
