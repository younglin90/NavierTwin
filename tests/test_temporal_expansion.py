"""Unsteady case expansion tests for mesh-native and grid operators."""

from __future__ import annotations

import numpy as np
import pytest

from naviertwin.core.preprocessing import expand_unsteady_case_snapshots


def test_unsteady_cases_expand_to_condition_time_samples() -> None:
    pv = pytest.importorskip("pyvista")
    from naviertwin.core.cfd_reader.base import CFDDataset

    cases = []
    for offset in (0.0, 10.0):
        mesh = pv.ImageData(dimensions=(4, 3, 1))
        series = np.stack(
            [np.full(mesh.n_points, offset + time) for time in (0.0, 0.5, 1.0)]
        )
        mesh.point_data["p"] = series[0]
        cases.append(
            CFDDataset(
                mesh=mesh,
                time_steps=[0.0, 0.5, 1.0],
                field_names=["p"],
                metadata={"time_series_fields": {"p": series}},
            )
        )

    expanded, params, names, has_time = expand_unsteady_case_snapshots(
        cases,
        [[1.0], [2.0]],
        ["Re"],
        field_names=["p"],
    )

    assert has_time
    assert len(expanded) == 6
    assert names == ["Re", "t"]
    np.testing.assert_allclose(
        params,
        [[1.0, 0.0], [1.0, 0.5], [1.0, 1.0], [2.0, 0.0], [2.0, 0.5], [2.0, 1.0]],
    )
    np.testing.assert_allclose(expanded[4].mesh.point_data["p"], 10.5)


def test_steady_cases_are_not_copied() -> None:
    case = type("Case", (), {"time_steps": [0.0]})()
    expanded, params, names, has_time = expand_unsteady_case_snapshots(
        [case], [[3.0]], ["Mach"], field_names=[]
    )

    assert expanded == [case]
    np.testing.assert_allclose(params, [[3.0]])
    assert names == ["Mach"]
    assert not has_time
