"""Round 71 — Pipeline HDF5 체크포인트 저장/복원."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest


class TestPipelineCheckpoint:
    def test_save_and_load(self, tmp_path: Path) -> None:
        pytest.importorskip("h5py")
        pytest.importorskip("sklearn")
        from naviertwin.core.digital_twin.pipeline import NavierTwinPipeline
        from naviertwin.core.digital_twin.pipeline_checkpoint import (
            load_pipeline_state,
            save_pipeline_state,
        )

        rng = np.random.default_rng(0)
        X = rng.standard_normal((30, 15))
        pipe = NavierTwinPipeline(reducer_kind="pod", n_modes=3, surrogate_kind="kriging")
        pipe.load_snapshots(X, field_name="U")
        pipe.reduce()
        params = np.linspace(0, 1, 15).reshape(-1, 1)
        pipe.fit_surrogate(params)
        pipe.validate(params[-3:], pipe.state.coeffs[-3:])

        out = tmp_path / "ckpt.h5"
        save_pipeline_state(pipe, out)
        assert out.exists()

        ckpt = load_pipeline_state(out)
        assert "snapshots" in ckpt
        assert "coeffs" in ckpt
        assert "modes" in ckpt
        assert ckpt["meta"]["reducer_kind"] == "pod"
        assert ckpt["meta"]["n_modes"] == 3
        assert "metrics" in ckpt
        assert "rmse" in ckpt["metrics"]

    def test_restore_roundtrip(self, tmp_path: Path) -> None:
        pytest.importorskip("h5py")
        pytest.importorskip("sklearn")
        from naviertwin.core.digital_twin.pipeline import NavierTwinPipeline
        from naviertwin.core.digital_twin.pipeline_checkpoint import (
            load_pipeline_state,
            restore_pipeline,
            save_pipeline_state,
        )

        rng = np.random.default_rng(0)
        X = rng.standard_normal((20, 10))
        p1 = NavierTwinPipeline(reducer_kind="pod", n_modes=3, surrogate_kind="rbf")
        p1.load_snapshots(X, field_name="U")
        p1.reduce()
        params = np.linspace(0, 1, 10).reshape(-1, 1)
        p1.fit_surrogate(params)

        out = tmp_path / "ckpt.h5"
        save_pipeline_state(p1, out)

        # 새 파이프라인에 복원
        p2 = NavierTwinPipeline(reducer_kind="pod", n_modes=3)
        ckpt = load_pipeline_state(out)
        restore_pipeline(p2, ckpt)

        assert p2.state.snapshots.shape == X.shape
        assert p2.state.reducer is not None
        assert p2.state.reducer.is_fitted

        # POD 모드로 복원 예측
        c = p2.state.reducer.encode(X)
        assert c.shape[0] == 10

    def test_missing_file_raises(self) -> None:
        pytest.importorskip("h5py")
        from naviertwin.core.digital_twin.pipeline_checkpoint import (
            load_pipeline_state,
        )

        with pytest.raises(FileNotFoundError):
            load_pipeline_state("/nonexistent.h5")

    def test_restore_incremental_pod_roundtrip(self, tmp_path: Path) -> None:
        pytest.importorskip("h5py")
        from naviertwin.core.digital_twin.pipeline import NavierTwinPipeline
        from naviertwin.core.digital_twin.pipeline_checkpoint import (
            load_pipeline_state,
            restore_pipeline,
            save_pipeline_state,
        )

        rng = np.random.default_rng(3)
        X = rng.standard_normal((24, 12))
        p1 = NavierTwinPipeline(reducer_kind="incremental_pod", n_modes=4, surrogate_kind="rbf")
        p1.load_snapshots(X, field_name="U")
        p1.reduce()

        out = tmp_path / "inc_ckpt.h5"
        save_pipeline_state(p1, out)

        p2 = NavierTwinPipeline(reducer_kind="incremental_pod", n_modes=4)
        ckpt = load_pipeline_state(out)
        restore_pipeline(p2, ckpt)

        assert p2.state.reducer is not None
        coeffs = p2.state.reducer.encode(X)
        assert coeffs.shape[0] == X.shape[1]

    def test_restore_mrpod_roundtrip(self, tmp_path: Path) -> None:
        pytest.importorskip("h5py")
        from naviertwin.core.digital_twin.pipeline import NavierTwinPipeline
        from naviertwin.core.digital_twin.pipeline_checkpoint import (
            load_pipeline_state,
            restore_pipeline,
            save_pipeline_state,
        )

        rng = np.random.default_rng(4)
        X = rng.standard_normal((18, 10))
        p1 = NavierTwinPipeline(reducer_kind="mrpod", n_modes=2, surrogate_kind="rbf")
        p1.load_snapshots(X, field_name="U")
        p1.reduce()

        out = tmp_path / "mr_ckpt.h5"
        save_pipeline_state(p1, out)

        p2 = NavierTwinPipeline(reducer_kind="mrpod", n_modes=2)
        ckpt = load_pipeline_state(out)
        restore_pipeline(p2, ckpt)

        assert p2.state.reducer is not None
        coeffs = p2.state.reducer.encode(X)
        assert coeffs.shape[0] == X.shape[1]
