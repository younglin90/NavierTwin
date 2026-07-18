"""R648 — PostProcessPanel GUI smoke tests."""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("PySide6")


@pytest.fixture
def app(qtbot):
    """qtbot fixture for Qt event loop."""
    return qtbot


class TestPanelConstruction:
    def test_panel_creates(self, qtbot) -> None:
        from naviertwin.gui.panels import PostProcessPanel

        panel = PostProcessPanel()
        qtbot.addWidget(panel)
        # op 리스트가 비어 있지 않음
        assert panel._op_list.count() > 0

    def test_category_filter(self, qtbot) -> None:
        from naviertwin.gui.panels import PostProcessPanel

        panel = PostProcessPanel()
        qtbot.addWidget(panel)
        n_total = panel._op_list.count()
        # 특정 카테고리로 변경
        panel._category_combo.setCurrentText("statistics")
        n_filtered = panel._op_list.count()
        assert 0 < n_filtered <= n_total

    def test_op_selection_updates_description(self, qtbot) -> None:
        from naviertwin.gui.panels.postproc_panel import PostProcessPanel

        panel = PostProcessPanel()
        qtbot.addWidget(panel)
        # 첫 op 선택
        panel._op_list.setCurrentRow(0)
        desc = panel._desc_label.text()
        assert len(desc) > 0
        assert "[" in desc

    def test_smoke_run_succeeds(self, qtbot) -> None:
        from naviertwin.gui.panels.postproc_panel import PostProcessPanel

        panel = PostProcessPanel()
        qtbot.addWidget(panel)

        # 모든 op에 대해 smoke 실행 시도
        for i in range(panel._op_list.count()):
            panel._op_list.setCurrentRow(i)
            op_name = panel._op_list.currentItem().text()
            panel._on_run_clicked()
            result_text = panel._result_text.toPlainText()
            assert "실행 실패" not in result_text, f"{op_name} failed: {result_text}"

    def test_signal_emitted(self, qtbot) -> None:
        from naviertwin.gui.panels.postproc_panel import PostProcessPanel

        panel = PostProcessPanel()
        qtbot.addWidget(panel)

        emitted: list = []
        panel.operation_done.connect(lambda name, res: emitted.append((name, res)))

        panel._op_list.setCurrentRow(0)
        panel._on_run_clicked()

        assert len(emitted) == 1
        name, result = emitted[0]
        assert isinstance(result, dict)

    def test_set_dataset_updates_customer_run_state(self, qtbot) -> None:
        from naviertwin.gui.panels.postproc_panel import PostProcessPanel

        panel = PostProcessPanel()
        qtbot.addWidget(panel)
        dataset = _FakeDataset()

        panel.set_dataset(dataset)

        assert panel._dataset is dataset
        assert "로드됨" in panel._data_label.text()
        assert panel._run_btn.text() == "실행 (로드 데이터)"

    def test_dataset_run_uses_loaded_fields(self, qtbot) -> None:
        from naviertwin.gui.panels.postproc_panel import PostProcessPanel

        panel = PostProcessPanel()
        qtbot.addWidget(panel)
        dataset = _FakeDataset()
        capture = _CaptureFacade()
        panel.set_dataset(dataset)
        _select_operation(panel, "psd_welch")
        panel._facade = capture

        panel._on_run_clicked()

        assert capture.calls
        op_name, kwargs = capture.calls[0]
        assert op_name == "psd_welch"
        np.testing.assert_allclose(kwargs["signal"], dataset.mesh.point_data["pressure"])
        assert "로드 데이터셋" in panel._result_text.toPlainText()


class TestSummarizeResult:
    def test_array_short(self) -> None:
        import numpy as np

        from naviertwin.gui.panels.postproc_panel import PostProcessPanel

        text = PostProcessPanel._summarize_result({"x": np.array([1.0, 2.0])})
        assert "1.0" in text

    def test_array_long_summarized(self) -> None:
        import numpy as np

        from naviertwin.gui.panels.postproc_panel import PostProcessPanel

        text = PostProcessPanel._summarize_result(
            {"x": np.zeros(100)},
        )
        assert "shape" in text and "mean" in text

    def test_dict_value(self) -> None:
        from naviertwin.gui.panels.postproc_panel import PostProcessPanel

        text = PostProcessPanel._summarize_result(
            {"box": {"median": 0.0, "Q1": -1.0, "Q3": 1.0}},
        )
        assert "keys" in text

    def test_scalar_value(self) -> None:
        from naviertwin.gui.panels.postproc_panel import PostProcessPanel

        text = PostProcessPanel._summarize_result({"r2": 0.95})
        assert "0.95" in text


class TestSmokeKwargs:
    def test_all_facade_ops_have_smoke_kwargs(self) -> None:
        from naviertwin.core.post_process_facade import PostProcessFacade
        from naviertwin.gui.panels.postproc_panel import PostProcessPanel

        facade = PostProcessFacade()
        for op_name in facade.list_operations():
            kwargs = PostProcessPanel._build_smoke_kwargs(op_name)
            assert isinstance(kwargs, dict)

    def test_undefined_op_raises(self) -> None:
        from naviertwin.gui.panels.postproc_panel import PostProcessPanel

        with pytest.raises(ValueError, match="smoke 데이터"):
            PostProcessPanel._build_smoke_kwargs("undefined_op_xyz")

    def test_dataset_kwargs_reject_unsupported_auto_mapping(self) -> None:
        from naviertwin.gui.panels.postproc_panel import PostProcessPanel

        with pytest.raises(ValueError, match="자동 구성"):
            PostProcessPanel._build_dataset_kwargs("surface_forces", _FakeDataset())

    def test_commercial_parity_modules_exposed_to_gui_facade(self) -> None:
        import inspect

        import naviertwin.core.post_process_facade as post_process_facade
        from naviertwin.core.post_process_facade import PostProcessFacade

        expected_modules = [
            "reynolds_stats",
            "psd",
            "surface_integrals",
            "quadrant_pdf",
            "two_point",
            "stat_convergence",
            "plane_flux",
            "time_interp",
            "coord_transform",
            "slice_extract",
            "expression_eval",
            "phase_lock",
            "running_moments",
            "denoise",
            "quantile_stats",
            "eof_analysis",
            "goodness_of_fit",
            "conditional_sampling",
            "grid_derivatives",
            "critical_points",
            "anisotropy",
            "morphology",
            "cell_volume",
            "truncation_criteria",
        ]
        source = inspect.getsource(post_process_facade)
        missing = [name for name in expected_modules if name not in source]

        assert missing == []
        assert len(PostProcessFacade().list_operations()) >= len(expected_modules)


class _FakeMesh:
    def __init__(self) -> None:
        self.n_points = 16
        self.n_cells = 4
        self.points = np.column_stack([
            np.linspace(0.0, 1.0, self.n_points),
            np.linspace(1.0, 2.0, self.n_points),
            np.zeros(self.n_points),
        ])
        self.point_data = {
            "pressure": np.linspace(100.0, 115.0, self.n_points),
            "velocity": np.column_stack([
                np.linspace(0.0, 1.0, self.n_points),
                np.linspace(1.0, 0.0, self.n_points),
                np.ones(self.n_points),
            ]),
        }
        self.cell_data = {}


class _FakeDataset:
    def __init__(self) -> None:
        self.mesh = _FakeMesh()
        self.field_names = ["pressure", "velocity"]
        self.time_steps = [0.0]
        self.metadata = {}

    @property
    def n_points(self) -> int:
        return self.mesh.n_points

    @property
    def n_cells(self) -> int:
        return self.mesh.n_cells

    @property
    def n_time_steps(self) -> int:
        return len(self.time_steps)

    def extract_field_snapshots(self, field_name: str) -> np.ndarray:
        values = np.asarray(self.mesh.point_data[field_name], dtype=float)
        return values.reshape(-1, 1)


class _CaptureFacade:
    def __init__(self) -> None:
        self.calls: list[tuple[str, dict[str, object]]] = []

    def run(self, op_name: str, **kwargs: object) -> dict[str, object]:
        self.calls.append((op_name, kwargs))
        return {"count": len(kwargs)}


def _select_operation(panel: object, op_name: str) -> None:
    for i in range(panel._op_list.count()):
        if panel._op_list.item(i).text() == op_name:
            panel._op_list.setCurrentRow(i)
            return
    raise AssertionError(f"operation not found: {op_name}")
