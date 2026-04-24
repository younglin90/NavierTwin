"""Round 76 — 배치 필드 예측."""

from __future__ import annotations

import numpy as np
import pytest


class TestBatchPredict:
    def test_serial_matches_single(self) -> None:
        pytest.importorskip("sklearn")
        from naviertwin.core.digital_twin.batch_predict import batch_predict_fields
        from naviertwin.core.digital_twin.pipeline import NavierTwinPipeline

        rng = np.random.default_rng(0)
        X = rng.standard_normal((20, 12))
        P = np.linspace(0, 1, 12).reshape(-1, 1)

        pipe = NavierTwinPipeline(reducer_kind="pod", n_modes=3, surrogate_kind="rbf")
        pipe.load_snapshots(X, field_name="U")
        pipe.reduce()
        pipe.fit_surrogate(P)

        new_P = np.linspace(0.2, 0.9, 50).reshape(-1, 1)
        fields = batch_predict_fields(pipe, new_P, chunk_size=16)
        assert fields.shape == (20, 50)

        # 단일 호출과 동일해야 함
        ref = pipe.predict_field(new_P)
        assert np.allclose(fields, ref, atol=1e-10)

    def test_threaded_matches_serial(self) -> None:
        pytest.importorskip("sklearn")
        from naviertwin.core.digital_twin.batch_predict import batch_predict_fields
        from naviertwin.core.digital_twin.pipeline import NavierTwinPipeline

        rng = np.random.default_rng(1)
        X = rng.standard_normal((15, 10))
        P = np.linspace(0, 1, 10).reshape(-1, 1)
        pipe = NavierTwinPipeline(reducer_kind="pod", n_modes=3, surrogate_kind="rbf")
        pipe.load_snapshots(X, field_name="U")
        pipe.reduce()
        pipe.fit_surrogate(P)

        new_P = np.linspace(0.1, 0.8, 30).reshape(-1, 1)
        f_serial = batch_predict_fields(pipe, new_P, chunk_size=10)
        f_threaded = batch_predict_fields(pipe, new_P, chunk_size=10, max_workers=2)
        assert np.allclose(f_serial, f_threaded, atol=1e-10)
