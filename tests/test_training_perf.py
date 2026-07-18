"""v5.6 P1 — GPU 효율 학습 경로(AMP·미니배치) + 디바이스 보고 테스트.

계약:
    - 기본값(batch_size=None, use_amp=False)은 기존 numerics 를 그대로 보존
      한다 — 같은 seed 로 두 번 학습하면 손실 이력이 완전히 동일해야 한다.
    - use_amp=True 는 CPU 에서 조용한 no-op (손실 이력이 기본 경로와 동일).
    - batch_size 를 주면 미니배치로 학습한다 — 손실은 유한하고 감소해야 한다.
    - device_used / training_metadata["device_used"] 는 실제 torch 디바이스
      문자열("cuda:0"/"cpu")을 보고한다 (UI 디바이스 배지용).
"""

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch", reason="학습 성능 테스트에는 torch 가 필요합니다.")
pytest.importorskip("pyvista", reason="데모 데이터셋 생성에는 pyvista 가 필요합니다.")

from naviertwin.core.operator_learning.fno.geometry_fno import (  # noqa: E402
    GeometryFNO2D,
)
from naviertwin.core.physnemo.cfd_field_model import (  # noqa: E402
    PhysicsNeMoCFDFieldModel,
)
from naviertwin.web.service import make_demo_dataset  # noqa: E402

_CUDA = torch.cuda.is_available()


# ----------------------------------------------------------------------
# 헬퍼
# ----------------------------------------------------------------------


@pytest.fixture(scope="module")
def demo():
    """작은 합성 시계열 데이터셋 (64 점 × 4 스텝 = 256 학습 행)."""
    return make_demo_dataset(nx=8, ny=8, n_steps=4)


def _fit_cfd(demo, **overrides) -> PhysicsNeMoCFDFieldModel:
    options: dict = {
        "hidden": 8,
        "n_layers": 2,
        "max_epochs": 30,
        "device": "cpu",
        "seed": 0,
    }
    options.update(overrides)
    model = PhysicsNeMoCFDFieldModel(**options)
    model.fit_datasets([demo], field_names=["p"])
    return model


def _make_fno_tensors(n_cases: int = 4, h: int = 8, w: int = 8):
    """작은 합성 (N,H,W,3)→(N,H,W,1) 케이스 — 채널의 선형결합이라 학습 가능."""
    rng = np.random.default_rng(0)
    x = rng.standard_normal((n_cases, h, w, 3)).astype(np.float32)
    y = (x[..., :1] - 0.5 * x[..., 1:2] + 0.25 * x[..., 2:3]).astype(np.float32)
    return x, y


def _fit_fno(**overrides) -> tuple[GeometryFNO2D, np.ndarray]:
    options: dict = {
        "n_params": 1,
        "modes": 4,
        "width": 8,
        "n_layers": 2,
        "epochs": 15,
        "device": "cpu",
        "seed": 0,
    }
    options.update(overrides)
    op = GeometryFNO2D(**options)
    x, y = _make_fno_tensors()
    op.fit(x, y)
    return op, x


# ----------------------------------------------------------------------
# (a)+(e) 기본 경로 결정성 + 강제 CPU 디바이스 보고
# ----------------------------------------------------------------------


def test_cfd_default_path_deterministic_and_reports_cpu(demo) -> None:
    """기본값으로 같은 seed 두 번 → 손실 이력 완전 동일 (numerics 불변 회귀)."""
    first = _fit_cfd(demo)
    second = _fit_cfd(demo)
    assert first.train_losses_ == second.train_losses_
    assert len(first.train_losses_) == 30
    # (e) device="cpu" 강제 → "cpu" 보고.
    assert first.device_used == "cpu"
    assert first.training_metadata["device_used"] == "cpu"
    assert first.training_metadata["use_amp"] is False
    assert first.training_metadata["batch_size"] is None


def test_fno_default_path_deterministic_and_reports_cpu() -> None:
    first, _ = _fit_fno()
    second, _ = _fit_fno()
    assert first.train_losses_ == second.train_losses_
    assert first.device_used == "cpu"
    assert first.training_metadata["device_used"] == "cpu"
    assert first.training_metadata["use_amp"] is False
    assert first.training_metadata["batch_size"] is None


def test_device_used_is_none_before_fit() -> None:
    assert PhysicsNeMoCFDFieldModel().device_used is None
    assert GeometryFNO2D(n_params=1).device_used is None


# ----------------------------------------------------------------------
# (b) 미니배치 경로가 학습된다
# ----------------------------------------------------------------------


def test_cfd_minibatch_trains(demo) -> None:
    default = _fit_cfd(demo)
    mini = _fit_cfd(demo, batch_size=64, max_epochs=40)
    # 미니배치는 full-batch 와 다른 경로 (손실 이력이 달라진다).
    assert mini.train_losses_[: len(default.train_losses_)] != default.train_losses_
    losses = np.asarray(mini.train_losses_)
    assert np.all(np.isfinite(losses))
    assert losses[-1] < losses[0]
    assert mini.training_metadata["batch_size"] == 64

    prediction = mini.predict(np.array([[0.5]]))
    assert prediction.shape == (64, 1)  # (n_locations * 1채널, n_rows)
    assert np.all(np.isfinite(prediction))


def test_fno_minibatch_trains() -> None:
    op, x = _fit_fno(batch_size=2, epochs=25)
    losses = np.asarray(op.train_losses_)
    assert np.all(np.isfinite(losses))
    assert losses[-1] < losses[0]
    assert op.training_metadata["batch_size"] == 2

    prediction = op.predict(x)
    assert prediction.shape == (4, 8, 8, 1)
    assert np.all(np.isfinite(prediction))


# ----------------------------------------------------------------------
# (c) use_amp=True 는 CPU 에서 무해한 no-op
# ----------------------------------------------------------------------


def test_cfd_amp_on_cpu_is_noop(demo) -> None:
    default = _fit_cfd(demo)
    amp = _fit_cfd(demo, use_amp=True)
    # CPU 에서는 AMP 가 켜지지 않으므로 손실 이력까지 완전히 같아야 한다.
    assert amp.train_losses_ == default.train_losses_
    assert amp.device_used == "cpu"
    assert amp.training_metadata["use_amp"] is False


def test_fno_amp_on_cpu_is_noop() -> None:
    default, _ = _fit_fno()
    amp, x = _fit_fno(use_amp=True)
    assert amp.train_losses_ == default.train_losses_
    assert amp.device_used == "cpu"
    assert amp.training_metadata["use_amp"] is False
    assert np.all(np.isfinite(amp.predict(x)))


# ----------------------------------------------------------------------
# (d) CUDA — AMP + 미니배치 학습 + 디바이스 배지 보고
# ----------------------------------------------------------------------


@pytest.mark.skipif(not _CUDA, reason="CUDA 가 없어 GPU AMP 테스트를 건너뜁니다.")
def test_cfd_cuda_amp_minibatch(demo) -> None:
    model = _fit_cfd(demo, device="cuda", use_amp=True, batch_size=64, max_epochs=40)
    assert model.device_used is not None
    assert model.device_used.startswith("cuda")
    assert str(model.training_metadata["device_used"]).startswith("cuda")
    assert model.training_metadata["use_amp"] is True
    losses = np.asarray(model.train_losses_)
    assert np.all(np.isfinite(losses))
    assert losses[-1] < losses[0]
    assert np.all(np.isfinite(model.predict(np.array([[0.5]]))))


@pytest.mark.skipif(not _CUDA, reason="CUDA 가 없어 GPU AMP 테스트를 건너뜁니다.")
def test_fno_cuda_amp_minibatch() -> None:
    op, x = _fit_fno(device="cuda", use_amp=True, batch_size=2, epochs=25)
    assert op.device_used is not None
    assert op.device_used.startswith("cuda")
    assert op.training_metadata["use_amp"] is True
    losses = np.asarray(op.train_losses_)
    assert np.all(np.isfinite(losses))
    assert losses[-1] < losses[0]
    assert np.all(np.isfinite(op.predict(x)))
