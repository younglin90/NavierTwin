"""R660 — Post-Tools 패널 i18n 검증 (KO + EN)."""

from __future__ import annotations

import pytest

pytest.importorskip("PySide6")


class TestPanelI18nKO:
    def test_default_language_ko_labels(self, qtbot) -> None:
        from naviertwin.gui.panels.postproc_panel import PostProcessPanel

        panel = PostProcessPanel()  # 기본 ko
        qtbot.addWidget(panel)
        # 카테고리 콤보 첫 항목
        assert panel._category_combo.itemText(0) in ("전체", "All")
        # preset combo는 op 선택 시 채워짐
        panel._op_list.setCurrentRow(0)
        assert panel._preset_combo.itemText(0) in ("(없음)", "(none)")


class TestPanelI18nEN:
    def test_english_labels(self, qtbot) -> None:
        from naviertwin.gui.panels.postproc_panel import PostProcessPanel
        from naviertwin.utils.i18n import Translator

        en_t = Translator(lang="en")
        panel = PostProcessPanel(translator=en_t)
        qtbot.addWidget(panel)
        assert panel._category_combo.itemText(0) == "All"
        # preset combo는 op 선택 시 채워짐
        panel._op_list.setCurrentRow(0)
        assert panel._preset_combo.itemText(0) == "(none)"
        assert panel._run_btn.text() == "Run Demo (synthetic data)"
        assert panel._save_chart_btn.text() == "Save Chart Image (PNG/SVG)"
        assert panel._save_preset_btn.text() == "Save Preset"

    def test_english_labels_run_smoke_still_works(self, qtbot) -> None:
        """라벨이 영어일 때도 op 실행은 정상."""
        from naviertwin.gui.panels.postproc_panel import PostProcessPanel
        from naviertwin.utils.i18n import Translator

        en_t = Translator(lang="en")
        panel = PostProcessPanel(translator=en_t)
        qtbot.addWidget(panel)
        panel._op_list.setCurrentRow(0)
        panel._on_run_clicked()
        assert "실행 실패" not in panel._result_text.toPlainText()


class TestI18nFallback:
    def test_translator_failure_uses_keys(self, qtbot, monkeypatch) -> None:
        """Translator import 실패 시 키 그대로 표시 (fallback)."""
        from naviertwin.gui.panels.postproc_panel import PostProcessPanel

        # callable 형 fallback
        panel = PostProcessPanel(translator=lambda key: key)
        qtbot.addWidget(panel)
        # 키가 그대로 라벨에 들어감
        assert panel._run_btn.text() == "posttools.run.demo"


class TestCategoryFilterAcrossLangs:
    def test_filter_with_korean(self, qtbot) -> None:
        from naviertwin.gui.panels.postproc_panel import PostProcessPanel

        panel = PostProcessPanel()
        qtbot.addWidget(panel)
        # statistics 카테고리로 변경
        panel._category_combo.setCurrentText("statistics")
        n = panel._op_list.count()
        assert n > 0

    def test_filter_with_english(self, qtbot) -> None:
        from naviertwin.gui.panels.postproc_panel import PostProcessPanel
        from naviertwin.utils.i18n import Translator

        panel = PostProcessPanel(translator=Translator(lang="en"))
        qtbot.addWidget(panel)
        panel._category_combo.setCurrentText("rom")
        n = panel._op_list.count()
        assert n > 0


class TestRegression:
    def test_existing_smoke_unchanged(self, qtbot) -> None:
        from naviertwin.gui.panels.postproc_panel import PostProcessPanel

        panel = PostProcessPanel()
        qtbot.addWidget(panel)
        for i in range(panel._op_list.count()):
            panel._op_list.setCurrentRow(i)
            panel._on_run_clicked()
            txt = panel._result_text.toPlainText()
            assert "실행 실패" not in txt
