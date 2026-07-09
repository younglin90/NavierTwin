"""AI Bench (벤치마크 데이터셋 → 연산자 학습) 테스트.

:mod:`naviertwin.web.bench` 서비스 계층과 웹 앱의 ⑧ AI Bench 콜백을 검증한다.
내장 솔버 생성 / PDEBench 포맷 HDF5 로드 / FNO 학습·평가 전 구간을 작은
문제 크기로 headless 실행한다.
"""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("torch", reason="AI Bench 테스트에는 torch 가 필요합니다.")

from naviertwin.web import bench  # noqa: E402

# 작은 문제 크기 (CI 속도).
_N_SAMPLES = 10
_N_X = 32
_EPOCHS = 6


@pytest.fixture(scope="module")
def heat_dataset():
    return bench.generate_operator_dataset("heat", n_samples=_N_SAMPLES, n_x=_N_X, n_steps=100)


@pytest.fixture(scope="module")
def trained(heat_dataset):
    return bench.train_operator(heat_dataset, epochs=_EPOCHS, modes=8, width=16)


# ──────────────────────────────────────────────────────────────────────
# 데이터셋 생성 / 로드
# ──────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize("kind", ["burgers", "heat"])
def test_generate_operator_dataset_shapes(kind) -> None:
    ds = bench.generate_operator_dataset(kind, n_samples=6, n_x=24, n_steps=80)
    assert ds["inputs"].shape == (6, 24, 1)
    assert ds["outputs"].shape == (6, 24, 1)
    assert ds["x"].shape == (24,)
    assert np.isfinite(ds["inputs"]).all()
    assert np.isfinite(ds["outputs"]).all()
    assert ds["source"] == f"builtin:{kind}"
    assert kind in bench.dataset_summary(ds)


def test_generate_cavity2d_timestep_pairs() -> None:
    ds = bench.generate_operator_dataset("cavity2d", n_samples=8, n_x=20)
    # (B, ny, nx, 2) 시간전진 페어, 채널 [ux, uy].
    assert ds["inputs"].shape == (8, 20, 20, 2)
    assert ds["outputs"].shape == (8, 20, 20, 2)
    assert np.isfinite(ds["inputs"]).all()
    assert ds["kind"] == "cavity2d"
    # 시간전진: 입력과 출력이 동일하면 안 된다 (유동이 진화).
    assert not np.allclose(ds["inputs"], ds["outputs"])


def test_generate_rejects_unknown_kind() -> None:
    with pytest.raises(ValueError):
        bench.generate_operator_dataset("navier_stokes_3d")


def test_load_pdebench_hdf5_roundtrip(tmp_path) -> None:
    import h5py

    path = tmp_path / "pdebench_burgers.h5"
    data = np.random.default_rng(0).random((9, 5, 40)).astype(np.float32)
    with h5py.File(path, "w") as f:
        f.create_dataset("tensor", data=data)
        f.create_dataset("x-coordinate", data=np.linspace(0.0, 1.0, 40))

    ds = bench.load_pdebench_hdf5(path, max_samples=6)
    assert ds["inputs"].shape == (6, 40, 1)
    assert ds["outputs"].shape == (6, 40, 1)
    assert ds["kind"] == "pdebench"
    # u0 = 첫 timestep, uT = 마지막 timestep.
    np.testing.assert_allclose(ds["inputs"][0, :, 0], data[0, 0, :], rtol=1e-6)
    np.testing.assert_allclose(ds["outputs"][0, :, 0], data[0, -1, :], rtol=1e-6)


def test_load_pdebench_finds_unnamed_tensor(tmp_path) -> None:
    import h5py

    path = tmp_path / "other.h5"
    with h5py.File(path, "w") as f:
        grp = f.create_group("data")
        grp.create_dataset("u", data=np.random.rand(4, 3, 16))

    ds = bench.load_pdebench_hdf5(path)
    assert ds["inputs"].shape == (4, 16, 1)


def test_load_pdebench_rejects_bad_files(tmp_path) -> None:
    import h5py

    with pytest.raises(FileNotFoundError):
        bench.load_pdebench_hdf5(tmp_path / "missing.h5")
    with pytest.raises(ValueError):
        bench.load_pdebench_hdf5(tmp_path)  # 확장자 아님(디렉토리)

    empty = tmp_path / "empty.h5"
    with h5py.File(empty, "w") as f:
        f.create_dataset("scalar", data=np.zeros(5))  # 3D 텐서 없음
    with pytest.raises(ValueError):
        bench.load_pdebench_hdf5(empty)


# ──────────────────────────────────────────────────────────────────────
# 연산자 학습 / 평가
# ──────────────────────────────────────────────────────────────────────


def test_train_operator_learns_and_reports(trained) -> None:
    losses = trained["losses"]
    assert len(losses) == _EPOCHS
    assert losses[-1] < losses[0]  # 손실 감소
    assert trained["n_train"] + trained["n_test"] == _N_SAMPLES
    assert trained["test_rmse"] >= 0
    assert trained["latency_ms"] > 0
    assert trained["train_time_s"] > 0
    assert getattr(trained["model"], "is_fitted", False)


def test_train_operator_rejects_unknown_operator(heat_dataset) -> None:
    with pytest.raises(ValueError):
        bench.train_operator(heat_dataset, operator="transformer")


def test_train_operator_requires_samples() -> None:
    tiny = {
        "inputs": np.zeros((2, 16, 1)),
        "outputs": np.zeros((2, 16, 1)),
    }
    with pytest.raises(ValueError):
        bench.train_operator(tiny)


def test_train_operator_progress_cb(heat_dataset) -> None:
    """progress_cb 가 epoch 1..N 를 순서대로, (epoch, total, loss) 로 호출한다."""
    calls: list[tuple[int, int, float]] = []
    bench.train_operator(
        heat_dataset,
        epochs=5,
        modes=8,
        width=12,
        progress_cb=lambda e, n, loss: calls.append((e, n, loss)),
    )
    assert [c[0] for c in calls] == [1, 2, 3, 4, 5]  # epoch 1-index 순서
    assert all(c[1] == 5 for c in calls)  # total 일관
    assert all(np.isfinite(c[2]) for c in calls)  # loss 유한


def test_evaluate_sample_shapes_and_clamp(trained, heat_dataset) -> None:
    ev = bench.evaluate_sample(trained["model"], heat_dataset, 3)
    assert ev["index"] == 3
    assert ev["is_2d"] is False
    assert len(ev["x"]) == len(ev["true"]) == len(ev["pred"]) == _N_X
    assert np.isfinite(ev["pred"]).all()
    assert ev["latency_ms"] > 0
    # 범위 밖 인덱스는 clamp.
    ev2 = bench.evaluate_sample(trained["model"], heat_dataset, 999)
    assert ev2["index"] == _N_SAMPLES - 1


def test_train_and_evaluate_fno2d_auto() -> None:
    """cavity2d 4D 데이터셋 → auto 디스패치로 FNO2D 학습/2D 평가."""
    ds = bench.generate_operator_dataset("cavity2d", n_samples=8, n_x=16)
    result = bench.train_operator(ds, epochs=3, modes=6, width=8)
    assert result["operator"] == "fno2d"
    assert len(result["losses"]) == 3
    assert result["latency_ms"] > 0

    ev = bench.evaluate_sample(result["model"], ds, 1)
    assert ev["is_2d"] is True
    assert np.asarray(ev["true2d"]).shape == (16, 16)
    assert np.asarray(ev["pred2d"]).shape == (16, 16)
    assert np.asarray(ev["err2d"]).shape == (16, 16)


def test_train_operator_dim_mismatch(heat_dataset) -> None:
    # 3D 데이터에 fno2d 강제 → 차원 오류.
    with pytest.raises(ValueError):
        bench.train_operator(heat_dataset, operator="fno2d")


# ──────────────────────────────────────────────────────────────────────
# 웹 앱 콜백 (⑧ AI Bench)
# ──────────────────────────────────────────────────────────────────────


def _make_app(name: str):
    pytest.importorskip("trame", reason="앱 콜백 테스트에는 trame 이 필요합니다.")
    pytest.importorskip("pyvista", reason="앱 콜백 테스트에는 pyvista 가 필요합니다.")
    from trame.app import get_server

    from naviertwin.web.app import NavierTwinWebApp

    return NavierTwinWebApp(server=get_server(name, client_type="vue3"))


def test_app_bench_workflow() -> None:
    app = _make_app("nt-bench-flow")
    st = app.server.state

    st.nt_bench_kind = "heat"
    st.nt_bench_nsamples = _N_SAMPLES
    st.nt_bench_nx = _N_X
    app.bench_generate()
    assert st.nt_error == ""
    assert st.nt_bench_ready is True
    assert st.nt_bench_max_sample == _N_SAMPLES - 1
    assert "heat" in st.nt_bench_summary

    st.nt_bench_epochs = _EPOCHS
    st.nt_bench_modes = 8
    st.nt_bench_width = 16
    app.bench_train()
    assert st.nt_error == ""
    assert st.nt_bench_trained is True
    assert "RMSE" in st.nt_bench_train_summary
    assert st.nt_chart_img.startswith("data:image/png;base64,")
    assert "FNO 학습 손실" in st.nt_chart_title

    st.nt_bench_sample = 2
    app.bench_evaluate()
    assert st.nt_error == ""
    assert "샘플 #2" in st.nt_chart_title


def test_app_bench_pdebench_load(tmp_path) -> None:
    import h5py

    app = _make_app("nt-bench-h5")
    st = app.server.state
    path = tmp_path / "pb.h5"
    with h5py.File(path, "w") as f:
        f.create_dataset("tensor", data=np.random.rand(6, 4, 24))
    st.nt_bench_path = str(path)
    app.bench_load_h5()
    assert st.nt_error == ""
    assert st.nt_bench_ready is True
    assert "pdebench" in st.nt_bench_summary


def test_app_bench_guards() -> None:
    app = _make_app("nt-bench-guards")
    st = app.server.state
    app.bench_train()
    assert st.nt_error  # 데이터셋 없음
    st.nt_error = ""
    app.bench_evaluate()
    assert st.nt_error  # 모델 없음
    st.nt_error = ""
    app.bench_load_h5()
    assert st.nt_error  # 경로 없음
