"""Round 7 — loss curve / compare widget / i18n 통합 테스트."""

from __future__ import annotations

import pytest


class TestLossCurveWidget:
    def test_can_be_imported(self) -> None:
        pytest.importorskip("PySide6")
        from naviertwin.gui.widgets import LossCurveWidget

        assert LossCurveWidget is not None


class TestI18nKeysCoverage:
    """중요 UI 키가 ko/en 양쪽에 모두 존재하는지 확인."""

    def test_all_keys_present_in_both_langs(self) -> None:
        from naviertwin.utils.i18n import Translator

        t_ko = Translator(lang="ko")
        t_en = Translator(lang="en")
        required = [
            "panel.import", "panel.analyze", "panel.reduce",
            "panel.model", "panel.twin", "panel.export",
            "metric.rmse", "metric.r2",
            "button.train", "button.run",
        ]
        for k in required:
            assert t_ko(k) != k, f"ko 번역 누락: {k}"
            assert t_en(k) != k, f"en 번역 누락: {k}"


class TestPipelineIntegrationWithMetrics:
    """파이프라인 결과가 비교 대시보드 dict 형태로 변환되는지."""

    def test_pipeline_metrics_dict_shape(self) -> None:
        pytest.importorskip("sklearn")
        import numpy as np

        from naviertwin.core.digital_twin.pipeline import NavierTwinPipeline

        rng = np.random.default_rng(0)
        X = rng.standard_normal((30, 15))

        results: dict[str, dict[str, float]] = {}
        for name, sk in [("Kriging", "kriging"), ("RBF", "rbf")]:
            p = NavierTwinPipeline(reducer_kind="pod", n_modes=3, surrogate_kind=sk)
            p.load_snapshots(X, field_name="U")
            p.reduce()
            params = np.linspace(0, 1, 15).reshape(-1, 1)
            p.fit_surrogate(params)
            metrics = p.validate(params[-4:], p.state.coeffs[-4:])
            results[name] = {"rmse": metrics["rmse"], "r2": metrics["r2"]}

        assert set(results.keys()) == {"Kriging", "RBF"}
        for v in results.values():
            assert "rmse" in v and "r2" in v
            assert v["rmse"] >= 0
