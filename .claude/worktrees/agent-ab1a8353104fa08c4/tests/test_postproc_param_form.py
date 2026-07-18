"""R654 — Post-Tools 패널 동적 scalar 파라미터 폼 검증."""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("PySide6")


class TestParamFormBasics:
    def test_form_built_for_psd_welch(self, qtbot) -> None:
        from naviertwin.gui.panels.postproc_panel import PostProcessPanel

        panel = PostProcessPanel()
        qtbot.addWidget(panel)
        # psd_welch 선택 → fs/nperseg/window 폼 위젯 생성
        panel._on_op_selected("psd_welch")
        assert "fs" in panel._param_widgets
        assert "nperseg" in panel._param_widgets
        assert "window" in panel._param_widgets

    def test_form_empty_for_no_scalar_params(self, qtbot) -> None:
        from naviertwin.gui.panels.postproc_panel import PostProcessPanel

        panel = PostProcessPanel()
        qtbot.addWidget(panel)
        # gof_normality는 scalar 파라미터 없음
        panel._on_op_selected("gof_normality")
        # 폼 비어있음
        assert len(panel._param_widgets) == 0

    def test_form_rebuild_clears_old(self, qtbot) -> None:
        from naviertwin.gui.panels.postproc_panel import PostProcessPanel

        panel = PostProcessPanel()
        qtbot.addWidget(panel)
        panel._on_op_selected("psd_welch")
        panel._on_op_selected("denoise")
        # 다른 op으로 바뀌면 폼이 새로 생성됨
        assert "fs" not in panel._param_widgets
        assert "window_length" in panel._param_widgets


class TestReadParamValues:
    def test_int_param_read(self, qtbot) -> None:
        from naviertwin.gui.panels.postproc_panel import PostProcessPanel

        panel = PostProcessPanel()
        qtbot.addWidget(panel)
        panel._on_op_selected("denoise")
        panel._param_widgets["window_length"].setValue(21)
        panel._param_widgets["polyorder"].setValue(5)
        vals = panel._read_param_values()
        assert vals["window_length"] == 21
        assert vals["polyorder"] == 5

    def test_float_param_read(self, qtbot) -> None:
        from naviertwin.gui.panels.postproc_panel import PostProcessPanel

        panel = PostProcessPanel()
        qtbot.addWidget(panel)
        panel._on_op_selected("kolmogorov_slope")
        panel._param_widgets["dx"].setValue(0.5)
        vals = panel._read_param_values()
        assert abs(vals["dx"] - 0.5) < 1e-12

    def test_str_param_read(self, qtbot) -> None:
        from naviertwin.gui.panels.postproc_panel import PostProcessPanel

        panel = PostProcessPanel()
        qtbot.addWidget(panel)
        panel._on_op_selected("change_points")
        panel._param_widgets["method"].setCurrentText("pelt")
        vals = panel._read_param_values()
        assert vals["method"] == "pelt"


class TestUserParamOverride:
    def test_scalar_override_applied_to_run(self, qtbot) -> None:
        from naviertwin.gui.panels.postproc_panel import PostProcessPanel

        panel = PostProcessPanel()
        qtbot.addWidget(panel)

        # eof를 n_modes=8로 실행
        panel._op_list.setCurrentRow(panel._op_list.findItems(
            "eof", __import__("PySide6.QtCore", fromlist=["Qt"]).Qt.MatchFlag.MatchExactly,
        )[0].listWidget().row(panel._op_list.findItems(
            "eof", __import__("PySide6.QtCore", fromlist=["Qt"]).Qt.MatchFlag.MatchExactly,
        )[0]))
        panel._param_widgets["n_modes"].setValue(8)

        kwargs, source = panel._build_run_kwargs("eof")
        assert kwargs["n_modes"] == 8

    def test_smoke_used_when_no_dataset(self, qtbot) -> None:
        from naviertwin.gui.panels.postproc_panel import PostProcessPanel

        panel = PostProcessPanel()
        qtbot.addWidget(panel)
        panel._on_op_selected("psd_welch")
        kwargs, source = panel._build_run_kwargs("psd_welch")
        # signal은 합성 데이터에서 옴
        assert source == "demo"
        assert isinstance(kwargs["signal"], np.ndarray)
        # scalar는 폼 기본값 (fs default = 1.0)
        assert isinstance(kwargs["fs"], float)


class TestFacadeScalarSpec:
    def test_facade_method_returns_dict(self) -> None:
        from naviertwin.core.post_process_facade import PostProcessFacade

        facade = PostProcessFacade()
        spec = facade.scalar_param_specs("psd_welch")
        assert "fs" in spec
        assert spec["fs"]["type"] == "float"

    def test_unknown_op_raises(self) -> None:
        from naviertwin.core.post_process_facade import PostProcessFacade

        with pytest.raises(KeyError):
            PostProcessFacade().scalar_param_specs("bogus")

    def test_describe_includes_scalar_params(self) -> None:
        from naviertwin.core.post_process_facade import PostProcessFacade

        info = PostProcessFacade().describe("denoise")
        assert "scalar_params" in info
        assert "window_length" in info["scalar_params"]


class TestSmokeRunStillPasses:
    def test_all_ops_still_smokeable_with_form(self, qtbot) -> None:
        from naviertwin.gui.panels.postproc_panel import PostProcessPanel

        panel = PostProcessPanel()
        qtbot.addWidget(panel)
        # 모든 op 선택 후 실행 (폼이 자동 적용됨)
        for i in range(panel._op_list.count()):
            panel._op_list.setCurrentRow(i)
            op_name = panel._op_list.currentItem().text()
            panel._on_run_clicked()
            txt = panel._result_text.toPlainText()
            assert "실행 실패" not in txt, f"{op_name} failed: {txt}"
