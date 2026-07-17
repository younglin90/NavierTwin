"""타임스텝 1개짜리 필드 추출 — 조용한 데이터 손실 회귀 방지.

``_to_single_snapshot`` 은 마지막 축을 벡터 성분으로 보고 norm 을 취한다. 그런데
스칼라 필드를 ``(1, n_points)``(= 스텝 1개 × 점 n개)로 저장하면 그 규칙이 **필드
전체를 값 하나로 뭉갰다**. 오류가 아니라 조용한 손실이라, ROM 이 n_features=1 로
"학습 성공" 하는 데까지 갔다 (형상 가변 케이스 세트에서 실제로 발생).
"""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("pyvista", reason="격자 구성에 pyvista 가 필요합니다.")

from naviertwin.core.cfd_reader.base import CFDDataset  # noqa: E402


def _dataset(fields: dict[str, np.ndarray], n_points_side: int = 5) -> CFDDataset:
    import pyvista as pv

    grid = pv.ImageData(dimensions=(n_points_side, n_points_side, 1))
    mesh_fields = {}
    for name, arr in fields.items():
        mesh_fields[name] = arr[0]
    for name, arr in mesh_fields.items():
        grid.point_data[name] = arr
    return CFDDataset(
        mesh=grid,
        time_steps=[0.0],
        field_names=list(fields),
        metadata={
            "source": "test",
            "time_series_fields": fields,
            "time_series_locations": {name: "point" for name in fields},
        },
    )


def test_single_step_scalar_keeps_every_point() -> None:
    """(1, n_points) 스칼라가 값 하나로 뭉개지면 안 된다."""
    n = 25
    values = np.linspace(1.0, 25.0, n)
    ds = _dataset({"p": values[None, :]})

    snaps = ds.extract_field_snapshots("p")
    assert snaps.shape == (n, 1), f"점 {n}개가 {snaps.shape} 로 뭉개졌습니다"
    assert np.allclose(snaps[:, 0], values)


def test_single_step_vector_becomes_magnitude_per_point() -> None:
    """(1, n_points, 3) 벡터는 점마다 크기 1개 — 점 수는 보존된다."""
    n = 25
    vel = np.zeros((1, n, 3))
    vel[0, :, 0] = 3.0
    vel[0, :, 1] = 4.0
    ds = _dataset({"U": vel})

    snaps = ds.extract_field_snapshots("U")
    assert snaps.shape == (n, 1)
    assert np.allclose(snaps[:, 0], 5.0)  # |(3,4,0)| = 5


def test_multi_step_still_works() -> None:
    """기존 다중 스텝 경로가 깨지지 않았는지 — 회귀 방지."""
    n, steps = 25, 4
    values = np.arange(steps * n, dtype=float).reshape(steps, n)
    ds = CFDDataset(
        mesh=_dataset({"p": values[:1]}).mesh,
        time_steps=[0.0, 1.0, 2.0, 3.0],
        field_names=["p"],
        metadata={
            "source": "test",
            "time_series_fields": {"p": values},
            "time_series_locations": {"p": "point"},
        },
    )
    snaps = ds.extract_field_snapshots("p")
    assert snaps.shape == (n, steps)
    assert np.allclose(snaps, values.T)


def test_rom_refuses_ragged_cases_instead_of_training_on_garbage() -> None:
    """점 수가 다른 케이스에 ROM 은 **거절**해야 한다 — 조용히 학습하면 안 된다.

    필드 추출이 값 하나로 뭉개던 시절엔 모든 케이스가 n_features=1 이 되어 크기
    검사를 통과했고, ROM 이 아무 의미 없는 모델을 학습해 "성공" 을 보고했다.
    """
    import pyvista as pv

    from naviertwin.web import service

    datasets = []
    for side in (5, 6):
        grid = pv.ImageData(dimensions=(side, side, 1))
        n = grid.n_points
        values = np.linspace(0.0, 1.0, n)
        grid.point_data["p"] = values
        datasets.append(
            CFDDataset(
                mesh=grid,
                time_steps=[0.0],
                field_names=["p"],
                metadata={
                    "source": "test",
                    "time_series_fields": {"p": values[None, :]},
                    "time_series_locations": {"p": "point"},
                },
            )
        )
    params = np.array([[1.0], [2.0]])
    with pytest.raises(ValueError, match="점 수가 달라"):
        service.build_twin_from_cases(datasets, "p", 2, params, param_names=["mu"])
