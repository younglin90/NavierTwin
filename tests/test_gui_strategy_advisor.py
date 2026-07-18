"""데스크톱 GUI v5.2 개선 — 전략 어드바이저 + 데모 데이터 메뉴.

웹 ②Model 카드와 같은 능력 레지스트리 판정을 데스크톱 Model 패널이 보여주고,
웹 데모 카탈로그(시계열)를 데스크톱 도구 메뉴에서 즉시 로드할 수 있어야 한다.
"""

from __future__ import annotations

import pytest

pytest.importorskip("PySide6", reason="데스크톱 GUI 테스트에는 PySide6 가 필요합니다.")
pytest.importorskip("pyvista", reason="데모 데이터에는 pyvista 가 필요합니다.")


def test_model_panel_advisor_reflects_dataset(qtbot) -> None:
    """시계열 로드 → ROM/DMD 가능 표시, 단일 스냅샷 → 불가 표시."""
    from naviertwin.gui.panels.model_panel import ModelPanel
    from naviertwin.web import service

    panel = ModelPanel()
    qtbot.addWidget(panel)

    panel.set_dataset(service.make_demo_dataset(nx=10, ny=10, n_steps=12))
    text = panel._strategy_advisor_label.text()
    assert "✅" in text and "축소+보간" in text
    assert "추천:" in text

    panel.set_dataset(service.make_demo_dataset(nx=8, ny=8, n_steps=1))
    text = panel._strategy_advisor_label.text()
    # 단일 스냅샷: 전부 불가여야 한다 (🚫 만 있고 ✅ 없음).
    assert "🚫" in text
    assert "✅" not in text


def test_main_window_demo_menu_loads_dataset(qtbot) -> None:
    """도구 메뉴 데모 로드 → 전체 패널에 데이터셋이 배선된다."""
    from naviertwin.gui.main_window import MainWindow

    win = MainWindow(confirm_on_close=False)
    qtbot.addWidget(win)

    win._load_demo_dataset("waves")
    # Model 패널까지 흘러들었는지 — 어드바이저가 판정을 표시한다.
    assert win._model_panel._dataset is not None
    assert win._model_panel._dataset.n_time_steps > 1
    advisor = win._model_panel._strategy_advisor_label.text()
    assert "동역학 예보" in advisor and "✅" in advisor


def test_twin_panel_reads_web_engine_param_names(qtbot) -> None:
    """웹 계열 엔진(param_names 키)도 데스크톱 Twin 패널이 이름/차원을 읽는다.

    데스크톱은 원래 "parameter_names" 만 읽어서, 웹에서 학습한 (μ, t) 엔진을
    데스크톱에 물리면 스핀박스 라벨/개수가 어긋났다 — 패리티 버그 수정 검증.
    """
    from naviertwin.gui.panels.twin_panel import TwinPanel
    from naviertwin.web import service

    sweep = service.make_demo_case_set("sweep_unsteady")
    result = service.build_parametric_dmd_twin(
        sweep["datasets"], "p", sweep["params"], param_names=sweep["param_names"]
    )
    engine = result["engine"]

    panel = TwinPanel()
    qtbot.addWidget(panel)
    panel.set_engine(engine)
    assert panel._param_names == ["inlet_velocity", "t"]
    assert len(panel._param_spins) == 2  # input_dim 이 스핀박스 수를 맞춘다


def test_demo_menu_exists_with_karman(qtbot) -> None:
    """메뉴에 카르만(실제 LBM) 데모가 노출된다 — 웹과 카탈로그 동기."""
    from naviertwin.gui.main_window import MainWindow

    win = MainWindow(confirm_on_close=False)
    qtbot.addWidget(win)
    labels: list[str] = []
    for action in win._tools_menu.actions():
        menu = action.menu()
        if menu is not None and "데모 데이터" in action.text():
            labels = [a.text() for a in menu.actions()]
    assert labels, "도구 메뉴에 '데모 데이터 로드' 서브메뉴가 없습니다"
    assert any("카르만" in label for label in labels)
