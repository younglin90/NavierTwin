"""R656 — Post-Process 결과 export + 일괄 실행기 검증."""

from __future__ import annotations

import json

import numpy as np
import pytest


class TestCSVExport:
    def test_table_like_result(self) -> None:
        from naviertwin.core.post_process_export import result_to_csv_text

        result = {
            "frequency": np.array([1.0, 2.0, 3.0]),
            "psd": np.array([0.1, 0.2, 0.3]),
        }
        txt = result_to_csv_text(result)
        assert "frequency" in txt
        assert "psd" in txt
        assert "1.0" in txt
        assert "0.3" in txt

    def test_scalar_only_result(self) -> None:
        from naviertwin.core.post_process_export import result_to_csv_text

        result = {"slope": -1.667, "r2": 0.99}
        txt = result_to_csv_text(result)
        assert "slope" in txt
        assert "0.99" in txt

    def test_padded_unequal_lengths(self) -> None:
        from naviertwin.core.post_process_export import result_to_csv_text

        result = {
            "a": np.array([1.0, 2.0, 3.0]),
            "b": np.array([10.0, 20.0]),
        }
        txt = result_to_csv_text(result)
        # 최장 길이 = 3
        assert txt.count("\n") >= 4  # header + 3 rows

    def test_save_csv(self, tmp_path) -> None:
        from naviertwin.core.post_process_export import save_csv

        result = {
            "x": np.array([1.0, 2.0]),
            "y": np.array([3.0, 4.0]),
        }
        path = tmp_path / "out.csv"
        p = save_csv(result, path)
        assert p.exists()
        assert "x,y" in p.read_text()


class TestJSONExport:
    def test_small_array_full(self, tmp_path) -> None:
        from naviertwin.core.post_process_export import save_json

        result = {"vals": np.array([1.0, 2.0, 3.0]), "n": 3}
        path = tmp_path / "out.json"
        save_json(result, path)
        data = json.loads(path.read_text())
        assert data["vals"] == [1.0, 2.0, 3.0]
        assert data["n"] == 3

    def test_large_array_summarized(self, tmp_path) -> None:
        from naviertwin.core.post_process_export import save_json

        result = {"big": np.zeros(2000)}
        path = tmp_path / "out.json"
        save_json(result, path)
        data = json.loads(path.read_text())
        assert data["big"]["_type"] == "ndarray"
        assert "summary" in data["big"]

    def test_nested_dict(self, tmp_path) -> None:
        from naviertwin.core.post_process_export import save_json

        result = {"box": {"median": 0.0, "iqr": 1.5}}
        path = tmp_path / "out.json"
        save_json(result, path)
        data = json.loads(path.read_text())
        assert data["box"]["median"] == 0.0

    def test_numpy_scalar(self, tmp_path) -> None:
        from naviertwin.core.post_process_export import save_json

        result = {"r2": np.float64(0.95), "n": np.int64(100)}
        path = tmp_path / "out.json"
        save_json(result, path)
        data = json.loads(path.read_text())
        assert abs(data["r2"] - 0.95) < 1e-12
        assert data["n"] == 100


class TestNPZExport:
    def test_basic_save_load(self, tmp_path) -> None:
        from naviertwin.core.post_process_export import save_npz

        result = {
            "modes": np.zeros((10, 5)),
            "sv": np.array([3.0, 2.0, 1.0]),
            "n": 5,
        }
        path = tmp_path / "out.npz"
        save_npz(result, path)
        assert path.exists()
        data = np.load(str(path))
        assert "modes" in data.files
        assert data["modes"].shape == (10, 5)
        assert int(data["n"]) == 5

    def test_no_data_raises(self, tmp_path) -> None:
        from naviertwin.core.post_process_export import save_npz

        with pytest.raises(ValueError, match="저장 가능한"):
            save_npz({"text": "hello"}, tmp_path / "x.npz")


class TestRunCategory:
    def test_run_statistics_category(self) -> None:
        from naviertwin.core.post_process_export import run_category
        from naviertwin.core.post_process_facade import PostProcessFacade

        facade = PostProcessFacade()
        rng = np.random.default_rng(0)
        smoke = {
            "reynolds_stats": {"u": rng.standard_normal((100, 5))},
            "box_stats": {"x": rng.standard_normal(500)},
            "quantile": {"x": rng.standard_normal(200), "q": 50.0},
        }
        results = run_category(facade, "statistics", smoke_kwargs=smoke)
        # 적어도 위 3개는 실행됨
        assert len(results) >= 3
        assert "reynolds_stats" in results

    def test_unknown_category_empty(self) -> None:
        from naviertwin.core.post_process_export import run_category
        from naviertwin.core.post_process_facade import PostProcessFacade

        results = run_category(PostProcessFacade(), "unknown_cat",
                               smoke_kwargs={})
        assert results == {}


class TestRunAll:
    def test_run_subset(self) -> None:
        from naviertwin.core.post_process_export import run_all
        from naviertwin.core.post_process_facade import PostProcessFacade

        facade = PostProcessFacade()
        smoke = {
            "psd_welch": {
                "signal": np.random.default_rng(0).standard_normal(500),
                "fs": 100.0,
                "nperseg": 64,
            },
            "quantile": {"x": np.arange(101.0), "q": 50.0},
        }
        results = run_all(facade, smoke)
        assert "psd_welch" in results
        assert "quantile" in results
        assert "_error" not in results["psd_welch"]
        assert "_error" not in results["quantile"]

    def test_run_with_failure(self) -> None:
        from naviertwin.core.post_process_export import run_all
        from naviertwin.core.post_process_facade import PostProcessFacade

        facade = PostProcessFacade()
        smoke = {
            "psd_welch": {"signal": np.array([1.0]), "fs": 1.0},  # 너무 짧음
        }
        results = run_all(facade, smoke)
        assert "psd_welch" in results
        assert "_error" in results["psd_welch"]


class TestBulkSummary:
    def test_markdown_format(self) -> None:
        from naviertwin.core.post_process_export import bulk_summary_markdown

        bulk = {
            "psd_welch": {
                "frequency": np.zeros(10),
                "psd": np.ones(10),
            },
            "failed_op": {"_error": "test failure"},
        }
        md = bulk_summary_markdown(bulk)
        assert "# Bulk Post-Process Summary" in md
        assert "## psd_welch" in md
        assert "✅" in md
        assert "❌" in md
        assert "총 op: 2" in md

    def test_empty_bulk(self) -> None:
        from naviertwin.core.post_process_export import bulk_summary_markdown

        md = bulk_summary_markdown({})
        assert "총 op: 0" in md
