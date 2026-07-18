"""R650 — auto post-process report generator."""

from __future__ import annotations

import numpy as np
import pytest


class TestAnalyzeProbeSignal:
    def test_basic_keys(self) -> None:
        from naviertwin.core.flow_analysis.auto_report import analyze_probe_signal

        rng = np.random.default_rng(0)
        u = np.sin(np.linspace(0, 4 * np.pi, 1000)) + 0.1 * rng.standard_normal(1000)
        report = analyze_probe_signal(u, fs=100.0)
        for k in ("n_samples", "fs", "duration", "features",
                   "box_stats", "psd", "convergence"):
            assert k in report

    def test_with_period_hint(self) -> None:
        from naviertwin.core.flow_analysis.auto_report import analyze_probe_signal

        t = np.linspace(0, 50, 5000)
        u = np.sin(2 * np.pi * t)  # period = 1
        report = analyze_probe_signal(u, fs=100.0, period_hint=1.0)
        assert "phase_lock" in report
        # 진폭 ~ 2 (max - min)
        assert report["phase_lock"]["phase_avg_amplitude"] > 1.5

    def test_change_points_disabled(self) -> None:
        from naviertwin.core.flow_analysis.auto_report import analyze_probe_signal

        rng = np.random.default_rng(1)
        u = rng.standard_normal(500)
        report = analyze_probe_signal(u, fs=10.0, detect_changes=False)
        assert "change_points" not in report

    def test_short_signal_raises(self) -> None:
        from naviertwin.core.flow_analysis.auto_report import analyze_probe_signal

        with pytest.raises(ValueError, match="too short"):
            analyze_probe_signal(np.zeros(10))

    def test_psd_peak_detection(self) -> None:
        from naviertwin.core.flow_analysis.auto_report import analyze_probe_signal

        # 5Hz 사인파 → PSD 피크가 5 근처
        t = np.linspace(0, 10, 2000)
        u = np.sin(2 * np.pi * 5 * t)
        report = analyze_probe_signal(u, fs=200.0)
        peak_f = report["psd"]["peak_frequency"]
        assert abs(peak_f - 5.0) < 1.0


class TestAnalyzeFieldSnapshots:
    def test_basic_keys(self) -> None:
        from naviertwin.core.flow_analysis.auto_report import analyze_field_snapshots

        rng = np.random.default_rng(2)
        X = rng.standard_normal((100, 30))
        report = analyze_field_snapshots(X, n_modes=5)
        for k in ("n_t", "n_x", "pod", "truncation", "statistics"):
            assert k in report

    def test_pod_energy_sum_in_range(self) -> None:
        from naviertwin.core.flow_analysis.auto_report import analyze_field_snapshots

        rng = np.random.default_rng(3)
        # rank 3 데이터 → 처음 3 모드가 에너지 대부분
        U = rng.standard_normal((50, 3))
        V = rng.standard_normal((3, 30))
        X = (U @ V).T  # (n_t=30, n_x=50)
        report = analyze_field_snapshots(X, n_modes=5)
        assert report["pod"]["cumulative_energy"] > 0.95

    def test_invalid_X_raises(self) -> None:
        from naviertwin.core.flow_analysis.auto_report import analyze_field_snapshots

        with pytest.raises(ValueError, match="2D"):
            analyze_field_snapshots(np.zeros(50))


class TestToMarkdown:
    def test_probe_report_markdown(self) -> None:
        from naviertwin.core.flow_analysis.auto_report import (
            analyze_probe_signal,
            to_markdown,
        )

        rng = np.random.default_rng(4)
        u = rng.standard_normal(500)
        report = analyze_probe_signal(u, fs=10.0)
        md = to_markdown(report)
        # 핵심 섹션이 있음
        assert "# Auto Post-Process Report" in md
        assert "## 메타" in md
        assert "## 시계열 특성" in md
        assert "## Power Spectral Density" in md

    def test_field_report_markdown(self) -> None:
        from naviertwin.core.flow_analysis.auto_report import (
            analyze_field_snapshots,
            to_markdown,
        )

        rng = np.random.default_rng(5)
        X = rng.standard_normal((50, 20))
        report = analyze_field_snapshots(X, n_modes=3)
        md = to_markdown(report)
        assert "POD Decomposition" in md
        assert "Spatial Statistics" in md

    def test_empty_report(self) -> None:
        from naviertwin.core.flow_analysis.auto_report import to_markdown

        md = to_markdown({})
        assert md.startswith("# Auto Post-Process Report")
