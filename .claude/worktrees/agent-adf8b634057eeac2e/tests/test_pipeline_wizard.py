"""Round 5 — 파이프라인 오케스트레이터 + 대시보드 + 위자드 테스트."""

from __future__ import annotations

import numpy as np
import pytest


class TestPipeline:
    def test_end_to_end_pod_kriging(self) -> None:
        pytest.importorskip("sklearn")
        from naviertwin.core.digital_twin.pipeline import NavierTwinPipeline

        rng = np.random.default_rng(0)
        # 저랭크 합성 스냅샷 (n_features, n_snapshots)
        r = 3
        U = rng.standard_normal((40, r))
        V = rng.standard_normal((r, 20))
        X = U @ V + 0.01 * rng.standard_normal((40, 20))

        pipe = NavierTwinPipeline(reducer_kind="pod", n_modes=r, surrogate_kind="kriging")
        pipe.load_snapshots(X, field_name="U")
        pipe.reduce()
        params = np.linspace(0, 1, 20).reshape(-1, 1)
        pipe.fit_surrogate(params)

        # 예측
        field = pipe.predict_field(np.array([[0.5]]))
        assert field.shape[0] == 40  # n_features

        # 검증
        metrics = pipe.validate(params[-5:], pipe.state.coeffs[-5:])
        assert "rmse" in metrics and "r2" in metrics

    def test_pipeline_export_report(self, tmp_path) -> None:
        pytest.importorskip("sklearn")
        pytest.importorskip("jinja2")
        from naviertwin.core.digital_twin.pipeline import NavierTwinPipeline

        rng = np.random.default_rng(0)
        X = rng.standard_normal((20, 10))
        pipe = NavierTwinPipeline(reducer_kind="pod", n_modes=3, surrogate_kind="rbf")
        pipe.load_snapshots(X, field_name="U")
        pipe.reduce()
        params = np.linspace(0, 1, 10).reshape(-1, 1)
        pipe.fit_surrogate(params)
        pipe.validate(params[:3], pipe.state.coeffs[:3])

        out = pipe.export_report(str(tmp_path / "report.html"), project="Test")
        content = out.read_text(encoding="utf-8")
        assert "POD" in content or "pod" in content.lower()

    def test_pipeline_missing_snapshots_raises(self) -> None:
        from naviertwin.core.digital_twin.pipeline import NavierTwinPipeline

        pipe = NavierTwinPipeline()
        with pytest.raises(RuntimeError, match="load"):
            pipe.reduce()

    def test_pipeline_unknown_reducer(self) -> None:
        from naviertwin.core.digital_twin.pipeline import NavierTwinPipeline

        pipe = NavierTwinPipeline(reducer_kind="foo")
        pipe.load_snapshots(np.zeros((5, 5)), field_name="U")
        with pytest.raises(ValueError, match="reducer"):
            pipe.reduce()

    def test_end_to_end_incremental_pod_rbf(self) -> None:
        pytest.importorskip("sklearn")
        from naviertwin.core.digital_twin.pipeline import NavierTwinPipeline

        rng = np.random.default_rng(1)
        n_modes = 3
        U = rng.standard_normal((30, n_modes))
        V = rng.standard_normal((n_modes, 16))
        X = U @ V + 0.01 * rng.standard_normal((30, 16))

        pipe = NavierTwinPipeline(
            reducer_kind="incremental_pod", n_modes=n_modes, surrogate_kind="rbf"
        )
        pipe.load_snapshots(X, field_name="U")
        pipe.reduce()
        params = np.linspace(0, 1, 16).reshape(-1, 1)
        pipe.fit_surrogate(params)
        field = pipe.predict_field(np.array([[0.5]]))
        assert field.shape[0] == 30

    def test_end_to_end_mrpod_rbf(self) -> None:
        pytest.importorskip("sklearn")
        from naviertwin.core.digital_twin.pipeline import NavierTwinPipeline

        rng = np.random.default_rng(2)
        n_modes = 2
        U = rng.standard_normal((28, n_modes))
        V = rng.standard_normal((n_modes, 14))
        X = U @ V + 0.01 * rng.standard_normal((28, 14))

        pipe = NavierTwinPipeline(reducer_kind="mrpod", n_modes=n_modes, surrogate_kind="rbf")
        pipe.load_snapshots(X, field_name="U")
        pipe.reduce()
        params = np.linspace(0, 1, 14).reshape(-1, 1)
        pipe.fit_surrogate(params)
        field = pipe.predict_field(np.array([[0.3]]))
        assert field.shape[0] == 28


class TestModelCompareWidget:
    def test_import_only(self) -> None:
        """PySide6 import 가능한지 + 클래스 정의만 확인."""
        pytest.importorskip("PySide6")
        from naviertwin.gui.widgets import ModelCompareWidget

        # 생성은 QApplication 필요 — 여기선 단순 import 확인으로 대체
        assert ModelCompareWidget is not None


class TestTutorialWizard:
    def test_run_pipeline_headless(self) -> None:
        """위자드의 run_pipeline 은 QWizard 인스턴스 없이 호출 불가.

        대신 핵심 파이프라인 흐름만 검증 (위자드 로직과 동일).
        """
        pytest.importorskip("sklearn")
        from naviertwin.core.digital_twin.pipeline import NavierTwinPipeline

        rng = np.random.default_rng(0)
        n_modes = 3
        U = rng.standard_normal((50, n_modes))
        V = rng.standard_normal((n_modes, 30))
        X = U @ V + 0.02 * rng.standard_normal((50, 30))

        pipe = NavierTwinPipeline(reducer_kind="pod", n_modes=n_modes)
        pipe.load_snapshots(X, field_name="U")
        pipe.reduce()
        params = np.linspace(0, 1, 30).reshape(-1, 1)
        pipe.fit_surrogate(params)
        metrics = pipe.validate(params[-5:], pipe.state.coeffs[-5:])
        assert metrics["rmse"] >= 0
