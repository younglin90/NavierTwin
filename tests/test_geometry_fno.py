"""GeometryFNO2D + 케이스 텐서라이저 (v5.2 형상 인지 FNO) 테스트.

합성 형상 가변 케이스(반지름이 다른 원기둥 구멍 + 포텐셜 유동 p)를 테스트
안에서 직접 만들어 검증한다 — 진리값을 해석적으로 알 수 있어 약한 일반화
검사(held-out 반지름)까지 정직하게 수행할 수 있다.
"""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("torch", reason="GeometryFNO2D 테스트에는 torch 가 필요합니다.")
pytest.importorskip("pyvista", reason="텐서라이저 테스트에는 pyvista 가 필요합니다.")
pytest.importorskip("scipy", reason="SDF(EDT) 계산에는 scipy 가 필요합니다.")

from naviertwin.core.operator_learning.fno.case_tensorizer import (  # noqa: E402
    cases_to_grid_tensors,
)
from naviertwin.core.operator_learning.fno.geometry_fno import (  # noqa: E402
    GeometryFNO2D,
)

_CENTER = (0.5, 0.5)
_N_SIDE = 32
_RESOLUTION = 32
# 학습 4개 + held-out 1개 (인덱스 3). held-out 0.22 의 최근접 학습 반지름은
# 0.24 (idx 4), 최원거리 학습 반지름은 0.08 (idx 0).
_RADII = [0.08, 0.12, 0.16, 0.22, 0.24]
_TRAIN_IDX = [0, 1, 2, 4]
_HELD_OUT_IDX = 3
_NEAREST_IDX = 4
_FARTHEST_IDX = 0


def _make_cylinder_case(radius: float, n_side: int = _N_SIDE):
    """원기둥 구멍(threshold) + 포텐셜 유동 p 를 갖는 합성 케이스를 만든다.

    web/service 의 "shapes" 데모와 같은 레시피 — 케이스마다 점/셀 수가 달라
    형상 가변 경로를 강제한다. p 는 반지름의 해석적 함수라 진리값 비교가 된다.
    """
    import pyvista as pv

    from naviertwin.core.cfd_reader.base import CFDDataset

    grid = pv.ImageData(
        dimensions=(n_side, n_side, 1),
        spacing=(1.0 / (n_side - 1), 1.0 / (n_side - 1), 1.0),
    )
    coords = np.asarray(grid.points, dtype=np.float64)
    center = np.asarray(_CENTER)
    dist = np.linalg.norm(coords[:, :2] - center, axis=1)
    unstructured = grid.cast_to_unstructured_grid()
    unstructured.point_data["dist"] = dist
    case = unstructured.threshold(radius, scalars="dist")  # 원기둥 제거
    pts = np.asarray(case.points, dtype=np.float64)
    case.point_data["p"] = _analytic_p(pts, radius)
    case.point_data.pop("dist", None)
    return CFDDataset(
        mesh=case,
        time_steps=[0.0],
        field_names=["p"],
        metadata={"source": "test_geometry_fno"},
    )


def _analytic_p(pts: np.ndarray, radius: float) -> np.ndarray:
    """원기둥 주위 포텐셜 유동 압력 (web/service "shapes" 데모와 동일식)."""
    center = np.asarray(_CENTER)
    rr = np.maximum(np.linalg.norm(pts[:, :2] - center, axis=1), 1e-6)
    theta = np.arctan2(pts[:, 1] - center[1], pts[:, 0] - center[0])
    ratio = radius / rr
    return 1.0 - (1.0 + ratio**4 - 2.0 * ratio**2 * np.cos(2.0 * theta))


@pytest.fixture(scope="module")
def tensorized():
    """반지름 5개 케이스를 한 번에 텐서화한다 (모듈 공유 — 비용 절약)."""
    cases = [_make_cylinder_case(r) for r in _RADII]
    params = np.asarray(_RADII, dtype=np.float64).reshape(-1, 1)
    return cases_to_grid_tensors(
        cases,
        params,
        field_names=["p"],
        resolution=_RESOLUTION,
        param_names=["radius"],
    )


# ──────────────────────────────────────────────────────────────────────
# 케이스 텐서라이저
# ──────────────────────────────────────────────────────────────────────


def test_tensorizer_shapes_and_channel_names(tensorized) -> None:
    inputs = tensorized["inputs"]
    targets = tensorized["targets"]
    grid = tensorized["grid"]
    meta = tensorized["meta"]

    n_cases = len(_RADII)
    assert inputs.ndim == 4 and inputs.shape[0] == n_cases
    assert inputs.shape[-1] == 3  # [sdf, mask, radius]
    assert targets.shape == (*inputs.shape[:3], 1)
    assert inputs.dtype == np.float32
    assert targets.dtype == np.float32
    assert tensorized["channel_names"] == ["sdf", "mask", "radius"]
    assert meta["target_names"] == ["p"]

    height, width = meta["hw"]
    assert inputs.shape[1:3] == (height, width)
    assert height * width == grid.n_points  # 두께 0 축 squeeze 후 격자와 일치
    assert 1 in tuple(grid.dimensions)  # 공통 격자도 2D
    assert np.isfinite(inputs).all()
    assert np.isfinite(targets).all()


def test_tensorizer_sdf_mask_structure(tensorized) -> None:
    """sdf 부호(구멍 안 − / 유체 +)와 mask(0/1) 구조를 기하로 직접 검증한다."""
    grid = tensorized["grid"]
    pts = np.asarray(grid.points, dtype=np.float64)
    dist = np.linalg.norm(pts[:, :2] - np.asarray(_CENTER), axis=1)
    # 공통 격자는 합집합 바운드에 2% 패딩을 더하므로 도메인 [0,1]² 밖 격자점은
    # mask=0 이다 — 유체 판정은 도메인 안쪽에서만 한다.
    in_domain = (
        (pts[:, 0] > 0.1) & (pts[:, 0] < 0.9) & (pts[:, 1] > 0.1) & (pts[:, 1] < 0.9)
    )

    for i, radius in enumerate(_RADII):
        # (H, W) 텐서는 평탄 VTK 순서를 reshape+squeeze 한 것 — ravel 로 복귀.
        sdf = tensorized["inputs"][i, :, :, 0].ravel()
        mask = tensorized["inputs"][i, :, :, 1].ravel()
        p = tensorized["targets"][i, :, :, 0].ravel()

        assert set(np.unique(mask)).issubset({0.0, 1.0})
        inside = dist < radius * 0.5  # 구멍 깊숙이
        outside = in_domain & (dist > radius + 0.15)  # 유체 확실 구역
        assert inside.any() and outside.any()
        assert mask[inside].max() == 0.0  # 구멍 안 = 고체
        assert mask[outside].min() == 1.0  # 먼 곳 = 유체
        assert sdf[inside].max() < 0.0  # 고체 안 부호거리 음수
        assert sdf[outside].min() > 0.0  # 유체 부호거리 양수
        # 파라미터 채널은 반지름 원시값 상수 브로드캐스트.
        assert np.allclose(tensorized["inputs"][i, :, :, 2], radius)
        # 고체 내부 타깃은 0 으로 채워진다.
        assert np.all(p[mask == 0.0] == 0.0)


def test_tensorizer_rejects_3d_cases() -> None:
    import pyvista as pv

    cube = pv.ImageData(dimensions=(6, 6, 6), spacing=(0.2, 0.2, 0.2))
    cube.point_data["p"] = np.zeros(cube.n_points)
    with pytest.raises(ValueError, match="2D"):
        cases_to_grid_tensors([cube], np.zeros((1, 1)), field_names=["p"])


def test_tensorizer_rejects_missing_field_and_bad_params(tensorized) -> None:
    cases = [_make_cylinder_case(0.1)]
    with pytest.raises(ValueError, match="필드"):
        cases_to_grid_tensors(cases, np.zeros((1, 1)), field_names=["ghost"])
    with pytest.raises(ValueError, match="params"):
        cases_to_grid_tensors(cases, np.zeros((3, 1)), field_names=["p"])
    with pytest.raises(ValueError, match="param_names"):
        cases_to_grid_tensors(
            cases, np.zeros((1, 2)), field_names=["p"], param_names=["only_one"]
        )
    with pytest.raises(ValueError, match="케이스"):
        cases_to_grid_tensors([], np.zeros((0, 1)), field_names=["p"])


def test_tensorizer_expands_vector_fields() -> None:
    """벡터 필드(U)는 성분별 채널(U_x, U_y, U_z)로 확장된다."""
    import pyvista as pv

    from naviertwin.core.cfd_reader.base import CFDDataset

    grid = pv.ImageData(dimensions=(12, 10, 1), spacing=(0.1, 0.1, 1.0))
    grid.point_data["U"] = np.random.default_rng(0).random((grid.n_points, 3))
    case = CFDDataset(mesh=grid, time_steps=[0.0], field_names=["U"])
    result = cases_to_grid_tensors(
        [case], np.zeros((1, 0)), field_names=["U"], resolution=12
    )
    assert result["meta"]["target_names"] == ["U_x", "U_y", "U_z"]
    assert result["targets"].shape[-1] == 3
    assert result["channel_names"] == ["sdf", "mask"]  # 파라미터 0개 허용


# ──────────────────────────────────────────────────────────────────────
# GeometryFNO2D
# ──────────────────────────────────────────────────────────────────────


def test_geometry_fno_trains_and_generalizes(tensorized) -> None:
    """소표본 4케이스 학습: 손실 감소 + held-out 반지름의 약한 일반화 검사.

    held-out r=0.22 예측은 최근접 학습 케이스(r=0.24)의 진리장보다 최원거리
    케이스(r=0.08)의 진리장에 가까우면 안 된다 — 합성 데이터의 정직한(약한)
    일반화 기준.
    """
    inputs = tensorized["inputs"]
    targets = tensorized["targets"]

    model = GeometryFNO2D(
        n_params=1,
        out_channels=1,
        modes=8,
        width=16,
        n_layers=3,
        epochs=60,
        lr=1e-3,
        seed=0,
    )
    model.fit(inputs[_TRAIN_IDX], targets[_TRAIN_IDX])

    assert model.is_fitted
    assert len(model.train_losses_) == 60
    assert model.train_losses_[-1] < model.train_losses_[0]  # 손실 감소
    assert np.isfinite(model.train_losses_).all()

    pred = model.predict(inputs[[_HELD_OUT_IDX]])
    assert pred.shape == (1, *inputs.shape[1:3], 1)
    assert np.isfinite(pred).all()

    err_near = float(np.sqrt(np.mean((pred[0] - targets[_NEAREST_IDX]) ** 2)))
    err_far = float(np.sqrt(np.mean((pred[0] - targets[_FARTHEST_IDX]) ** 2)))
    assert err_near < err_far

    # 배치 없는 입력 → 배치 없는 출력 (기존 FNO 계약과 같은 관용성).
    single = model.predict(inputs[_HELD_OUT_IDX])
    assert single.shape == (*inputs.shape[1:3], 1)
    np.testing.assert_allclose(single, pred[0], rtol=1e-5, atol=1e-5)


def test_geometry_fno_validates_contract(tensorized) -> None:
    inputs = tensorized["inputs"]
    targets = tensorized["targets"]

    with pytest.raises(ValueError, match="backend"):
        GeometryFNO2D(n_params=1, backend="bogus")
    with pytest.raises(ValueError, match="n_params"):
        GeometryFNO2D(n_params=-1)

    model = GeometryFNO2D(n_params=1, out_channels=1, modes=4, width=8, epochs=1)
    with pytest.raises(RuntimeError, match="fit"):
        model.predict(inputs)
    with pytest.raises(ValueError, match="채널"):
        model.fit(inputs[:, :, :, :2], targets)  # 파라미터 채널 누락
    with pytest.raises(ValueError, match="채널"):
        model.fit(inputs, np.repeat(targets, 2, axis=-1))  # 타깃 채널 초과
    with pytest.raises(ValueError, match="4D"):
        model.fit(inputs[0], targets[0])

    fitted = GeometryFNO2D(n_params=1, out_channels=1, modes=4, width=8, epochs=2)
    fitted.fit(inputs[_TRAIN_IDX], targets[_TRAIN_IDX])
    with pytest.raises(ValueError, match="채널"):
        fitted.predict(inputs[:, :, :, :2])


def test_geometry_fno_neuralop_backend(tensorized) -> None:
    """neuralop 백엔드 교체가 같은 계약(shape/손실)으로 동작한다."""
    pytest.importorskip("neuralop")
    inputs = tensorized["inputs"]
    targets = tensorized["targets"]

    model = GeometryFNO2D(
        n_params=1, out_channels=1, modes=4, width=8, n_layers=2,
        epochs=2, seed=0, backend="neuralop",
    )
    model.fit(inputs[_TRAIN_IDX], targets[_TRAIN_IDX])
    assert model.is_fitted
    assert len(model.train_losses_) == 2
    pred = model.predict(inputs[[_HELD_OUT_IDX]])
    assert pred.shape == (1, *inputs.shape[1:3], 1)
    assert pred.dtype == np.float32
    assert np.isfinite(pred).all()
