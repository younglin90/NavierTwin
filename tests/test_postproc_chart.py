"""R655 — Post-Process 결과 차트 위젯 검증."""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("PySide6")
pytest.importorskip("matplotlib")


class TestChartWidget:
    def test_create_chart(self, qtbot) -> None:
        from naviertwin.gui.widgets.postproc_chart import PostProcessChart

        chart = PostProcessChart()
        qtbot.addWidget(chart)
        assert chart._figure is not None
        assert chart._canvas is not None

    def test_render_psd(self, qtbot) -> None:
        from naviertwin.gui.widgets.postproc_chart import PostProcessChart

        chart = PostProcessChart()
        qtbot.addWidget(chart)
        f = np.linspace(0, 50, 100)
        P = np.exp(-f / 10) + 1e-3
        chart.render("psd_welch", {"frequency": f, "psd": P})
        assert chart._last_op == "psd_welch"

    def test_render_eof(self, qtbot) -> None:
        from naviertwin.gui.widgets.postproc_chart import PostProcessChart

        chart = PostProcessChart()
        qtbot.addWidget(chart)
        rng = np.random.default_rng(0)
        eofs = rng.standard_normal((40, 5))
        var = np.array([0.5, 0.3, 0.1, 0.05, 0.05])
        pcs = rng.standard_normal((30, 5))
        chart.render("eof", {"eofs": eofs, "var_explained": var, "pcs": pcs})
        assert chart._last_op == "eof"

    def test_render_box_stats(self, qtbot) -> None:
        from naviertwin.gui.widgets.postproc_chart import PostProcessChart

        chart = PostProcessChart()
        qtbot.addWidget(chart)
        result = {
            "box": {
                "median": 0.0,
                "Q1": -1.0,
                "Q3": 1.0,
                "iqr": 2.0,
                "whisker_low": -2.5,
                "whisker_high": 2.5,
                "n_outliers": 3,
            },
        }
        chart.render("box_stats", result)
        assert chart._last_op == "box_stats"

    def test_render_quadrant(self, qtbot) -> None:
        from naviertwin.gui.widgets.postproc_chart import PostProcessChart

        chart = PostProcessChart()
        qtbot.addWidget(chart)
        result = {"quadrants": {
            "Q1": {"fraction": 0.25, "count": 10},
            "Q2": {"fraction": 0.30, "count": 12},
            "Q3": {"fraction": 0.20, "count": 8},
            "Q4": {"fraction": 0.25, "count": 10},
            "hole": {"fraction": 0.0, "count": 0},
        }}
        chart.render("quadrant_analysis", result)

    def test_render_change_points(self, qtbot) -> None:
        from naviertwin.gui.widgets.postproc_chart import PostProcessChart

        chart = PostProcessChart()
        qtbot.addWidget(chart)
        chart.render("change_points", {
            "changepoints": [50, 120],
            "segment_means": [0.0, 5.0, 2.0],
        })

    def test_render_pod_truncation(self, qtbot) -> None:
        from naviertwin.gui.widgets.postproc_chart import PostProcessChart

        chart = PostProcessChart()
        qtbot.addWidget(chart)
        curve = np.cumsum(np.array([0.5, 0.3, 0.1, 0.05, 0.05]))
        chart.render("pod_truncation", {
            "n_modes": 3,
            "cumulative_energy": curve,
        })

    def test_render_two_point_acf(self, qtbot) -> None:
        from naviertwin.gui.widgets.postproc_chart import PostProcessChart

        chart = PostProcessChart()
        qtbot.addWidget(chart)
        r = np.linspace(0, 10, 50)
        R = np.exp(-r / 2)
        chart.render("two_point_acf", {"r": r, "R": R, "L_int": 2.0})

    def test_render_helmholtz(self, qtbot) -> None:
        from naviertwin.gui.widgets.postproc_chart import PostProcessChart

        chart = PostProcessChart()
        qtbot.addWidget(chart)
        rng = np.random.default_rng(0)
        chart.render("helmholtz_decomp", {
            "solenoidal_u": rng.standard_normal((16, 16)),
            "solenoidal_v": rng.standard_normal((16, 16)),
            "irrotational_u": rng.standard_normal((16, 16)),
            "irrotational_v": rng.standard_normal((16, 16)),
        })

    def test_render_critical_points(self, qtbot) -> None:
        from naviertwin.gui.widgets.postproc_chart import PostProcessChart

        chart = PostProcessChart()
        qtbot.addWidget(chart)
        chart.render("critical_points", {
            "critical_points": [
                {"x": 0.0, "y": 0.0, "type": "saddle"},
                {"x": 1.0, "y": 1.0, "type": "center"},
            ],
            "count": 2,
        })

    def test_render_subspace_drift(self, qtbot) -> None:
        from naviertwin.gui.widgets.postproc_chart import PostProcessChart

        chart = PostProcessChart()
        qtbot.addWidget(chart)
        chart.render("subspace_drift", {
            "angles": np.array([0.05, 0.1, 0.3, 0.5]),
            "drift_score": 0.32,
            "grassmann_distance": 0.6,
        })

    def test_render_morris(self, qtbot) -> None:
        from naviertwin.gui.widgets.postproc_chart import PostProcessChart

        chart = PostProcessChart()
        qtbot.addWidget(chart)
        chart.render("morris_sensitivity", {
            "mu_star": np.array([0.5, 1.0, 0.2]),
            "sigma": np.array([0.1, 0.3, 0.05]),
        })

    def test_fallback_for_unknown_op(self, qtbot) -> None:
        from naviertwin.gui.widgets.postproc_chart import PostProcessChart

        chart = PostProcessChart()
        qtbot.addWidget(chart)
        # 핸들러 없는 op도 fallback line plot
        chart.render("unknown_op", {"data": np.linspace(0, 1, 100)})
        assert chart._last_op == "unknown_op"

    def test_clear(self, qtbot) -> None:
        from naviertwin.gui.widgets.postproc_chart import PostProcessChart

        chart = PostProcessChart()
        qtbot.addWidget(chart)
        chart.render("psd_welch", {
            "frequency": np.linspace(0, 10, 50),
            "psd": np.exp(-np.linspace(0, 10, 50)),
        })
        chart.clear()
        assert chart._last_op is None

    def test_handler_failure_recovery(self, qtbot) -> None:
        from naviertwin.gui.widgets.postproc_chart import PostProcessChart

        chart = PostProcessChart()
        qtbot.addWidget(chart)
        # 차트 핸들러는 KeyError 발생하지만 위젯은 살아남음
        chart.render("psd_welch", {})  # frequency/psd 없음
        # 에러 메시지가 표시되었지만 예외 안 던짐
        assert chart._last_op == "psd_welch"


class TestPanelChartIntegration:
    def test_panel_has_chart_widget(self, qtbot) -> None:
        from naviertwin.gui.panels.postproc_panel import PostProcessPanel

        panel = PostProcessPanel()
        qtbot.addWidget(panel)
        assert panel._chart is not None

    def test_chart_updates_on_run(self, qtbot) -> None:
        from naviertwin.gui.panels.postproc_panel import PostProcessPanel

        panel = PostProcessPanel()
        qtbot.addWidget(panel)
        # psd_welch 선택 후 실행
        items = panel._op_list.findItems(
            "psd_welch",
            __import__("PySide6.QtCore", fromlist=["Qt"]).Qt.MatchFlag.MatchExactly,
        )
        if items:
            panel._op_list.setCurrentItem(items[0])
            panel._on_run_clicked()
            assert panel._chart._last_op == "psd_welch"
