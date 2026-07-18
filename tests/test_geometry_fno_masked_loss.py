"""GeometryFNO2D 마스킹 손실 (0-채움 영역 제외) 테스트.

외부 리뷰: 타깃의 고체/도메인 밖 격자점은 물리값이 아니라 자리채움용 0 이므로
손실에서 점수화하면 안 된다. 이 파일은

- 전부-1 마스크가 무마스크(빠른 경로)와 **비트 단위로** 같은지 (수치 보존),
- 큰 고체 영역이 0-채움된 경우 마스킹 학습이 유체 상수값을 훨씬 잘 복원하는지,
- 텐서라이저가 ``valid_mask`` 를 올바른 shape/값으로 내보내는지,
- neuralop 백엔드 + sample_masks 가 한국어 NotImplementedError 를 내는지

를 검증한다. 모든 학습은 결정성을 위해 device="cpu", 고정 시드로 돌린다.
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

# ──────────────────────────────────────────────────────────────────────
# (a) 전부-1 마스크 = 무마스크 (수치 보존)
# ──────────────────────────────────────────────────────────────────────


def test_all_ones_mask_matches_unmasked_bit_for_bit() -> None:
    """마스크가 전부 1 이면 손실 값·예측이 무마스크 경로와 완전히 일치한다.

    마스킹 MSE ``Σ(m·err²)/Σ_broadcast(m)`` 은 m 을 출력 채널로 브로드캐스트해
    합산 분모를 잡으므로 전부-1 에서 ``MSELoss`` (전체 평균)와 값·기울기가 같다.
    학습 루프(모델 구성·셔플 RNG 소비)도 FNO2D.fit 과 동일해 같은 시드에서
    예측이 비트 단위로 일치한다.
    """
    rng = np.random.default_rng(0)
    n, h, w = 4, 16, 16
    x = rng.standard_normal((n, h, w, 3)).astype(np.float32)
    y = rng.standard_normal((n, h, w, 1)).astype(np.float32)

    def _run(masks: np.ndarray | None) -> GeometryFNO2D:
        model = GeometryFNO2D(
            n_params=1, out_channels=1, modes=6, width=8, n_layers=2,
            epochs=15, lr=1e-3, seed=0, device="cpu",
        )
        model.fit(x, y, masks)
        return model

    unmasked = _run(None)
    masked = _run(np.ones((n, h, w), dtype=np.float32))

    # 손실 궤적이 동일 (마스킹 메타 플래그는 반대).
    np.testing.assert_allclose(
        masked.train_losses_, unmasked.train_losses_, rtol=0, atol=0
    )
    assert unmasked.training_metadata["masked_loss"] is False
    assert masked.training_metadata["masked_loss"] is True

    # 예측이 비트 단위로 일치.
    pred_unmasked = unmasked.predict(x)
    pred_masked = masked.predict(x)
    assert np.array_equal(pred_masked, pred_unmasked)


def test_mask_from_channel_defaults_off_and_derives_when_on() -> None:
    """mask_from_channel 기본 OFF 는 무마스크와 동일, ON 은 mask 채널을 쓴다."""
    rng = np.random.default_rng(3)
    n, h, w = 3, 12, 12
    x = rng.standard_normal((n, h, w, 3)).astype(np.float32)
    # mask 입력 채널(index 1)을 명시적 0/1 로 세팅.
    ch_mask = (rng.random((n, h, w)) > 0.3).astype(np.float32)
    x[..., 1] = ch_mask
    y = rng.standard_normal((n, h, w, 1)).astype(np.float32)

    off = GeometryFNO2D(
        n_params=1, modes=4, width=8, n_layers=2, epochs=6, seed=0, device="cpu"
    )
    off.fit(x, y)  # 기본 OFF → 무마스크
    assert off.training_metadata["masked_loss"] is False

    auto = GeometryFNO2D(
        n_params=1, modes=4, width=8, n_layers=2, epochs=6, seed=0, device="cpu"
    )
    auto.fit(x, y, mask_from_channel=True)  # mask 채널에서 유도
    assert auto.training_metadata["masked_loss"] is True

    explicit = GeometryFNO2D(
        n_params=1, modes=4, width=8, n_layers=2, epochs=6, seed=0, device="cpu"
    )
    explicit.fit(x, y, ch_mask)  # 같은 마스크를 명시 전달
    # mask_from_channel 유도 == 명시 전달 (동일 마스크 → 동일 결과).
    assert np.array_equal(auto.predict(x), explicit.predict(x))


# ──────────────────────────────────────────────────────────────────────
# (b) 큰 0-채움 고체 영역 — 마스킹이 유체 상수값을 더 잘 복원
# ──────────────────────────────────────────────────────────────────────


def test_masked_loss_recovers_fluid_constant_better() -> None:
    """유체=상수 c, 고체=0-채움 큰 영역에서 마스킹이 유체값을 훨씬 잘 맞춘다.

    무마스크 학습은 고체의 0 쪽으로 끌려가 유체 예측이 c 아래로 편향된다.
    마스킹 학습은 유체만 점수화해 c 를 복원한다 — 유체 영역 오차가 무마스크의
    절반 미만이어야 한다 (고체 비율을 크게, epoch 를 적게 잡아 편향을 부각).
    """
    n, h, w = 6, 24, 24
    c = 5.0
    # 유체는 오른쪽 25% 열에만 (고체 비율 0.75).
    mask = np.zeros((n, h, w), dtype=np.float32)
    mask[:, :, int(w * 0.75):] = 1.0
    solid_fraction = 1.0 - float(mask.mean())
    assert solid_fraction > 0.7  # 대비를 강하게

    x = np.zeros((n, h, w, 3), dtype=np.float32)
    y = np.zeros((n, h, w, 1), dtype=np.float32)
    for i in range(n):
        x[i, :, :, 0] = np.linspace(-1.0, 1.0, w)[None, :]  # sdf 유사 신호
        x[i, :, :, 1] = mask[i]                              # mask 채널
        x[i, :, :, 2] = 0.1 * i                              # 파라미터
        y[i, :, :, 0] = c * mask[i]                          # 유체=c, 고체=0

    def _run(masks: np.ndarray | None) -> GeometryFNO2D:
        model = GeometryFNO2D(
            n_params=1, out_channels=1, modes=6, width=12, n_layers=2,
            epochs=40, lr=3e-3, seed=0, device="cpu",
        )
        model.fit(x, y, masks)
        return model

    fluid = mask[0] > 0.5

    def _fluid_rmse(model: GeometryFNO2D) -> float:
        pred = model.predict(x)[..., 0]
        return float(np.sqrt(np.mean(((pred - c)[:, fluid]) ** 2)))

    err_unmasked = _fluid_rmse(_run(None))
    err_masked = _fluid_rmse(_run(mask))

    assert err_masked < 0.5 * err_unmasked


# ──────────────────────────────────────────────────────────────────────
# (c) 텐서라이저 valid_mask 키
# ──────────────────────────────────────────────────────────────────────


def _make_holed_case(radius: float, n_side: int = 24):
    """원형 구멍(threshold)을 가진 합성 케이스 — 고체 내부에서 mask=0 이 나온다."""
    import pyvista as pv

    from naviertwin.core.cfd_reader.base import CFDDataset

    grid = pv.ImageData(
        dimensions=(n_side, n_side, 1),
        spacing=(1.0 / (n_side - 1), 1.0 / (n_side - 1), 1.0),
    )
    coords = np.asarray(grid.points, dtype=np.float64)
    center = np.asarray((0.5, 0.5))
    dist = np.linalg.norm(coords[:, :2] - center, axis=1)
    ug = grid.cast_to_unstructured_grid()
    ug.point_data["dist"] = dist
    case = ug.threshold(radius, scalars="dist")
    case.point_data["p"] = np.asarray(case.points, dtype=np.float64)[:, 0]
    case.point_data.pop("dist", None)
    return CFDDataset(mesh=case, time_steps=[0.0], field_names=["p"])


def test_tensorizer_exposes_valid_mask() -> None:
    """valid_mask 가 (N, H, W) float32 0/1 로 나오고 mask 입력 채널과 일치한다."""
    radii = [0.12, 0.2]
    cases = [_make_holed_case(r) for r in radii]
    params = np.asarray(radii, dtype=np.float64).reshape(-1, 1)
    result = cases_to_grid_tensors(
        cases, params, field_names=["p"], resolution=24, param_names=["radius"]
    )

    valid_mask = result["valid_mask"]
    inputs = result["inputs"]

    assert valid_mask.shape == inputs.shape[:3]
    assert valid_mask.dtype == np.float32
    assert set(np.unique(valid_mask)).issubset({0.0, 1.0})
    # mask 입력 채널(index 1)과 정확히 같은 배열이어야 한다 (재사용, 재계산 아님).
    np.testing.assert_array_equal(valid_mask, inputs[:, :, :, 1])
    # 구멍 때문에 0 과 1 이 모두 존재해야 유효한 검증이 된다.
    assert (valid_mask == 0.0).any() and (valid_mask == 1.0).any()
    # 타깃은 0-채움 영역에서 정확히 0 이다.
    targets = result["targets"]
    assert np.all(targets[valid_mask == 0.0] == 0.0)


def test_tensorizer_valid_mask_feeds_masked_fit() -> None:
    """텐서라이저 valid_mask 를 그대로 fit(sample_masks=...) 에 넘길 수 있다."""
    radii = [0.12, 0.2, 0.16]
    cases = [_make_holed_case(r) for r in radii]
    params = np.asarray(radii, dtype=np.float64).reshape(-1, 1)
    result = cases_to_grid_tensors(
        cases, params, field_names=["p"], resolution=24, param_names=["radius"]
    )

    model = GeometryFNO2D(
        n_params=1, out_channels=1, modes=4, width=8, n_layers=2,
        epochs=3, seed=0, device="cpu",
    )
    model.fit(result["inputs"], result["targets"], result["valid_mask"])
    assert model.is_fitted
    assert model.training_metadata["masked_loss"] is True
    pred = model.predict(result["inputs"])
    assert pred.shape == result["targets"].shape
    assert np.isfinite(pred).all()


# ──────────────────────────────────────────────────────────────────────
# (d) neuralop 백엔드 + 마스킹 → NotImplementedError
# ──────────────────────────────────────────────────────────────────────


def test_neuralop_backend_rejects_sample_masks() -> None:
    """neuralop 백엔드는 마스킹을 지원하지 않아 한국어 NotImplementedError 를 낸다."""
    rng = np.random.default_rng(2)
    n, h, w = 3, 12, 12
    x = rng.standard_normal((n, h, w, 3)).astype(np.float32)
    y = rng.standard_normal((n, h, w, 1)).astype(np.float32)
    mask = np.ones((n, h, w), dtype=np.float32)

    model = GeometryFNO2D(
        n_params=1, out_channels=1, modes=4, width=8, n_layers=2,
        epochs=1, seed=0, backend="neuralop",
    )
    with pytest.raises(NotImplementedError, match="neuralop"):
        model.fit(x, y, mask)
    # mask_from_channel 경로도 동일하게 막힌다.
    with pytest.raises(NotImplementedError, match="마스킹"):
        model.fit(x, y, mask_from_channel=True)
