"""CFDDataset.extract_field_snapshots 테스트."""

from __future__ import annotations

import numpy as np
import pytest

from naviertwin.core.cfd_reader.base import CFDDataset


def test_extract_single_snapshot(sample_mesh: object) -> None:
    """단일 스냅샷 필드는 (n_features, 1)로 반환되어야 한다."""
    dataset = CFDDataset(
        mesh=sample_mesh,
        time_steps=[0.0],
        field_names=["p"],
        metadata={},
    )
    snaps = dataset.extract_field_snapshots("p")
    assert snaps.shape == (sample_mesh.n_points, 1)


def test_extract_multi_timestep_scalar(sample_mesh: object) -> None:
    """타임스텝으로 누적된 스칼라 필드는 (n_points, n_steps)로 복원되어야 한다."""
    n_steps = 3
    base = np.asarray(sample_mesh.point_data["p"], dtype=float)
    stacked = np.concatenate([base, base + 1.0, base + 2.0], axis=0)
    sample_mesh.point_data["p_stacked"] = stacked

    dataset = CFDDataset(
        mesh=sample_mesh,
        time_steps=[0.0, 0.1, 0.2],
        field_names=["p_stacked"],
        metadata={},
    )
    snaps = dataset.extract_field_snapshots("p_stacked")
    assert snaps.shape == (sample_mesh.n_points, n_steps)


def test_extract_multi_timestep_vector(sample_mesh: object) -> None:
    """타임스텝으로 누적된 벡터 필드는 크기 기반 (n_points, n_steps)로 복원된다."""
    n_steps = 2
    base = np.asarray(sample_mesh.point_data["U"], dtype=float)
    stacked = np.vstack([base, base * 2.0])
    sample_mesh.point_data["u_stacked"] = stacked

    dataset = CFDDataset(
        mesh=sample_mesh,
        time_steps=[0.0, 0.1],
        field_names=["u_stacked"],
        metadata={},
    )
    snaps = dataset.extract_field_snapshots("u_stacked")
    assert snaps.shape == (sample_mesh.n_points, n_steps)


def test_extract_unknown_field_raises(sample_mesh: object) -> None:
    """없는 필드 요청 시 ValueError가 발생해야 한다."""
    dataset = CFDDataset(
        mesh=sample_mesh,
        time_steps=[0.0],
        field_names=[],
        metadata={},
    )
    with pytest.raises(ValueError):
        dataset.extract_field_snapshots("does_not_exist")
