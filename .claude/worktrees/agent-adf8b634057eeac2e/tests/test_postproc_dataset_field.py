"""R661-662 — dataset 필드 콤보 + 언어 실시간 전환 검증."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

pytest.importorskip("PySide6")


def _make_fake_dataset() -> SimpleNamespace:
    """가짜 CFDDataset 비슷한 객체 (mesh.point_data 포함)."""
    rng = np.random.default_rng(0)
    point_data = SimpleNamespace()
    # dict-like: items() 메서드
    pd_dict = {
        "U": rng.standard_normal(50),
        "p": rng.standard_normal(50),
        "T": rng.standard_normal(50),
    }
    point_data.items = lambda: pd_dict.items()
    mesh = SimpleNamespace(point_data=point_data, cell_data=None)
    return SimpleNamespace(
        mesh=mesh,
        field_names=list(pd_dict.keys()),
        n_points=50,
        n_cells=0,
    )


class TestFieldCombo:
    def test_disabled_initially(self, qtbot) -> None:
        from naviertwin.gui.panels.postproc_panel import PostProcessPanel

        panel = PostProcessPanel()
        qtbot.addWidget(panel)
        assert not panel._field_combo.isEnabled()

    def test_enabled_after_set_dataset(self, qtbot) -> None:
        from naviertwin.gui.panels.postproc_panel import PostProcessPanel

        panel = PostProcessPanel()
        qtbot.addWidget(panel)
        ds = _make_fake_dataset()
        panel.set_dataset(ds)
        assert panel._field_combo.isEnabled()
        names = [
            panel._field_combo.itemText(i)
            for i in range(panel._field_combo.count())
        ]
        # 자동 선택 + 3개 필드
        assert "(자동 선택)" in names
        assert "U" in names
        assert "p" in names

    def test_clear_dataset_resets(self, qtbot) -> None:
        from naviertwin.gui.panels.postproc_panel import PostProcessPanel

        panel = PostProcessPanel()
        qtbot.addWidget(panel)
        panel.set_dataset(_make_fake_dataset())
        panel.clear_dataset()
        assert not panel._field_combo.isEnabled()
        assert panel._dataset is None

    def test_selected_field_returns_none_for_auto(self, qtbot) -> None:
        from naviertwin.gui.panels.postproc_panel import PostProcessPanel

        panel = PostProcessPanel()
        qtbot.addWidget(panel)
        panel.set_dataset(_make_fake_dataset())
        # 기본값: "(자동 선택)"
        assert panel.selected_field_name() is None

    def test_selected_field_returns_chosen(self, qtbot) -> None:
        from naviertwin.gui.panels.postproc_panel import PostProcessPanel

        panel = PostProcessPanel()
        qtbot.addWidget(panel)
        panel.set_dataset(_make_fake_dataset())
        panel._field_combo.setCurrentText("p")
        assert panel.selected_field_name() == "p"

    def test_dataset_kwargs_uses_selected_field(self, qtbot) -> None:
        from naviertwin.gui.panels.postproc_panel import PostProcessPanel

        panel = PostProcessPanel()
        qtbot.addWidget(panel)
        ds = _make_fake_dataset()
        panel.set_dataset(ds)
        # T 선택
        panel._field_combo.setCurrentText("T")
        kwargs = PostProcessPanel._build_dataset_kwargs(
            "psd_welch", ds, preferred_field="T",
        )
        # signal은 T 필드 (50 pt)
        assert "signal" in kwargs
        assert kwargs["signal"].shape == (50,)

    def test_invalid_field_falls_back(self, qtbot) -> None:
        from naviertwin.gui.panels.postproc_panel import PostProcessPanel

        panel = PostProcessPanel()
        qtbot.addWidget(panel)
        ds = _make_fake_dataset()
        panel.set_dataset(ds)
        # 존재하지 않는 필드
        kwargs = PostProcessPanel._build_dataset_kwargs(
            "psd_welch", ds, preferred_field="bogus",
        )
        assert "signal" in kwargs


class TestLanguageLiveSwitch:
    def test_set_language_updates_buttons(self, qtbot) -> None:
        from naviertwin.gui.panels.postproc_panel import PostProcessPanel

        panel = PostProcessPanel()  # ko default
        qtbot.addWidget(panel)
        before = panel._run_btn.text()
        panel.set_language("en")
        after = panel._run_btn.text()
        assert "Run Demo" in after
        assert before != after

    def test_set_language_back_to_ko(self, qtbot) -> None:
        from naviertwin.gui.panels.postproc_panel import PostProcessPanel

        panel = PostProcessPanel()
        qtbot.addWidget(panel)
        panel.set_language("en")
        panel.set_language("ko")
        assert "Demo" in panel._run_btn.text() or "실행" in panel._run_btn.text()

    def test_retranslate_ui_no_crash_without_translator(self, qtbot) -> None:
        from naviertwin.gui.panels.postproc_panel import PostProcessPanel

        panel = PostProcessPanel(translator=lambda key: key)
        qtbot.addWidget(panel)
        panel.retranslate_ui()  # 예외 안 남
        # set_language도 안전하게 noop
        panel.set_language("en")

    def test_panel_directly_translator_swap(self, qtbot) -> None:
        """패널의 Translator를 직접 교체 + retranslate."""
        from naviertwin.gui.panels.postproc_panel import PostProcessPanel
        from naviertwin.utils.i18n import Translator

        panel = PostProcessPanel(translator=Translator(lang="ko"))
        qtbot.addWidget(panel)
        ko_text = panel._run_btn.text()
        panel._t = Translator(lang="en")
        panel.retranslate_ui()
        en_text = panel._run_btn.text()
        assert "Run" in en_text
        assert ko_text != en_text


class TestRegression:
    def test_existing_smoke_passes(self, qtbot) -> None:
        from naviertwin.gui.panels.postproc_panel import PostProcessPanel

        panel = PostProcessPanel()
        qtbot.addWidget(panel)
        for i in range(panel._op_list.count()):
            panel._op_list.setCurrentRow(i)
            panel._on_run_clicked()
            txt = panel._result_text.toPlainText()
            assert "실행 실패" not in txt
