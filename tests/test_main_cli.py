"""Round 592 — main.py CLI path coverage."""

from __future__ import annotations

import json
import sys

import numpy as np
import pytest


class TestBuildParser:
    def test_parser_created(self) -> None:
        from naviertwin.main import _build_parser

        p = _build_parser()
        assert p.prog == "naviertwin"

    def test_parse_no_args(self) -> None:
        from naviertwin.main import _build_parser

        p = _build_parser()
        args = p.parse_args([])
        assert args.gui is False
        assert args.command is None

    def test_parse_gui_flag(self) -> None:
        from naviertwin.main import _build_parser

        p = _build_parser()
        args = p.parse_args(["--gui"])
        assert args.gui is True

    def test_parse_benchmark_subcommand(self) -> None:
        from naviertwin.main import _build_parser

        p = _build_parser()
        args = p.parse_args(["benchmark", "--kind", "burgers"])
        assert args.command == "benchmark"
        assert args.kind == "burgers"

    def test_parse_server_subcommand(self) -> None:
        from naviertwin.main import _build_parser

        p = _build_parser()
        args = p.parse_args(["server", "--port", "9000"])
        assert args.command == "server"
        assert args.port == 9000

    def test_parse_pipeline_subcommand(self) -> None:
        from naviertwin.main import _build_parser

        p = _build_parser()
        args = p.parse_args(["pipeline", "--reducer", "ae", "--n-modes", "8"])
        assert args.command == "pipeline"
        assert args.reducer == "ae"
        assert args.n_modes == 8

    def test_parse_model_sweep_subcommand(self) -> None:
        from naviertwin.main import _build_parser

        p = _build_parser()
        args = p.parse_args(
            [
                "model-sweep",
                "--reducers",
                "pod",
                "--n-modes",
                "2,4",
                "--surrogates",
                "rbf",
                "--json",
            ]
        )
        assert args.command == "model-sweep"
        assert args.reducers == "pod"
        assert args.n_modes == "2,4"
        assert args.surrogates == "rbf"
        assert args.as_json is True

    def test_parse_build_twin_subcommand(self, tmp_path) -> None:
        from naviertwin.main import _build_parser

        p = _build_parser()
        args = p.parse_args(
            [
                "build-twin",
                "--csv-snapshots",
                "snap_*.csv",
                "--field-column",
                "U",
                "--outdir",
                str(tmp_path),
                "--validation-count",
                "2",
                "--json",
            ]
        )
        assert args.command == "build-twin"
        assert args.csv_snapshots == "snap_*.csv"
        assert args.field_column == "U"
        assert args.validation_count == 2
        assert args.as_json is True

    def test_parse_predict_twin_subcommand(self, tmp_path) -> None:
        from naviertwin.main import _build_parser

        p = _build_parser()
        args = p.parse_args(
            [
                "predict-twin",
                "--engine",
                str(tmp_path / "engine.pkl"),
                "--params",
                "0.25",
                "--json",
            ]
        )
        assert args.command == "predict-twin"
        assert args.engine.endswith("engine.pkl")
        assert args.params == "0.25"
        assert args.as_json is True

    def test_parse_validate_twin_subcommand(self, tmp_path) -> None:
        from naviertwin.main import _build_parser

        p = _build_parser()
        args = p.parse_args(
            [
                "validate-twin",
                "--engine",
                str(tmp_path / "engine.pkl"),
                "--csv-snapshots",
                str(tmp_path / "snapshots"),
                "--field-column",
                "U",
                "--max-rmse",
                "0.1",
                "--json",
            ]
        )
        assert args.command == "validate-twin"
        assert args.engine.endswith("engine.pkl")
        assert args.csv_snapshots.endswith("snapshots")
        assert args.field_column == "U"
        assert args.max_rmse == 0.1
        assert args.as_json is True

    def test_parse_package_twin_subcommand(self, tmp_path) -> None:
        from naviertwin.main import _build_parser

        p = _build_parser()
        args = p.parse_args(
            [
                "package-twin",
                "--artifacts-dir",
                str(tmp_path / "twin"),
                "--output",
                str(tmp_path / "twin.zip"),
                "--json",
            ]
        )
        assert args.command == "package-twin"
        assert args.artifacts_dir.endswith("twin")
        assert args.output.endswith("twin.zip")
        assert args.as_json is True

    def test_parse_autorefine_subcommand(self) -> None:
        from naviertwin.main import _build_parser

        p = _build_parser()
        args = p.parse_args(["autorefine", "--interval-sec", "60", "--iterations", "2"])
        assert args.command == "autorefine"
        assert args.interval_sec == 60
        assert args.iterations == 2


class TestRunGui:
    def test_run_gui_no_pyside6(self, monkeypatch) -> None:
        import builtins

        from naviertwin.main import _run_gui

        real_import = builtins.__import__

        def block(name, *a, **kw):
            if name == "PySide6.QtWidgets":
                raise ImportError("blocked")
            return real_import(name, *a, **kw)

        monkeypatch.setattr(builtins, "__import__", block)
        code = _run_gui(None)
        assert code == 1


class TestRunBenchmark:
    def test_run_benchmark_unknown_kind(self) -> None:
        from naviertwin.main import _run_benchmark

        code = _run_benchmark("unknown_kind")
        assert code == 1


class TestRunServer:
    def test_run_server_no_uvicorn(self, monkeypatch) -> None:
        import builtins

        from naviertwin.main import _run_server

        real_import = builtins.__import__

        def block(name, *a, **kw):
            if name == "uvicorn":
                raise ImportError("blocked")
            return real_import(name, *a, **kw)

        monkeypatch.setattr(builtins, "__import__", block)
        code = _run_server("127.0.0.1", 8000)
        assert code == 1


class TestRunPipeline:
    def test_run_pipeline_pod_kriging(self) -> None:
        from naviertwin.main import _run_pipeline

        code = _run_pipeline("pod", 3, "kriging")
        assert code == 0

    def test_run_pipeline_ae_rbf(self) -> None:
        from naviertwin.main import _run_pipeline

        code = _run_pipeline("ae", 2, "rbf")
        assert code == 0


class TestRunModelSweep:
    def test_run_model_sweep_json(self, capsys) -> None:
        pytest.importorskip("sklearn")
        from naviertwin.main import _run_model_sweep

        code = _run_model_sweep(
            reducers="pod",
            n_modes="2,3",
            surrogates="rbf",
            samples=14,
            features=18,
            seed=3,
            as_json=True,
        )
        payload = json.loads(capsys.readouterr().out)

        assert code == 0
        assert payload["status"] == "ok"
        assert payload["configs"] == 2
        assert payload["rows"][0]["rmse"] <= payload["rows"][1]["rmse"]

    def test_run_model_sweep_invalid_config(self, capsys) -> None:
        from naviertwin.main import _run_model_sweep

        code = _run_model_sweep(
            reducers="unknown",
            n_modes="2",
            surrogates="rbf",
            samples=14,
            features=18,
            seed=3,
            as_json=True,
        )
        output = capsys.readouterr()

        assert code == 2
        assert output.out == ""
        assert "model-sweep error:" in output.err


class TestRunBuildTwin:
    def test_run_build_twin_from_csv_snapshots(self, tmp_path, capsys) -> None:
        pytest.importorskip("h5py")
        pytest.importorskip("pandas")
        pytest.importorskip("sklearn")
        from naviertwin.main import (
            _run_build_twin,
            _run_package_twin,
            _run_predict_twin,
            _run_validate_twin,
        )

        paths = []
        for step in range(10):
            path = tmp_path / f"snapshot_{step:03d}.csv"
            rows = ["x,U"]
            for index in range(8):
                rows.append(f"{index},{step * 0.2 + index * 0.01}")
            path.write_text("\n".join(rows) + "\n", encoding="utf-8")
            paths.append(path)

        code = _run_build_twin(
            input_path=None,
            csv_snapshots=",".join(str(path) for path in paths),
            field=None,
            field_column="U",
            params=None,
            param_columns=None,
            outdir=str(tmp_path / "twin"),
            reducer="pod",
            n_modes=2,
            surrogate="rbf",
            validation_count=2,
            as_json=True,
        )
        payload = json.loads(capsys.readouterr().out)
        from naviertwin.core.digital_twin.twin_engine import TwinEngine

        assert code == 0
        assert payload["status"] == "ok"
        assert payload["artifacts"]["checkpoint"].endswith("pipeline.h5")
        assert payload["artifacts"]["engine"].endswith("engine.pkl")
        assert payload["training"]["train_count"] == 8
        assert "rmse" in payload["metrics"]

        from naviertwin.utils.hashing import hash_file

        manifest = json.loads((tmp_path / "twin" / "manifest.json").read_text(encoding="utf-8"))
        integrity = manifest["extra"]["artifact_integrity"]
        assert set(integrity) == {"metrics", "checkpoint", "engine", "report"}
        assert integrity["engine"]["sha256"] == hash_file(tmp_path / "twin" / "engine.pkl")
        assert integrity["metrics"]["bytes"] > 0

        engine = TwinEngine.load(tmp_path / "twin" / "engine.pkl")
        prediction = engine.predict(np.array([0.25]))
        assert prediction.shape == (8,)

        predict_code = _run_predict_twin(
            engine_path=str(tmp_path / "twin" / "engine.pkl"),
            params="0.25",
            params_csv=None,
            param_columns=None,
            output=str(tmp_path / "prediction.csv"),
            as_json=True,
        )
        predict_payload = json.loads(capsys.readouterr().out)

        assert predict_code == 0
        assert predict_payload["prediction_shape"] == [8]
        assert (tmp_path / "prediction.csv").exists()

        validate_code = _run_validate_twin(
            engine_path=str(tmp_path / "twin" / "engine.pkl"),
            input_path=None,
            csv_snapshots=",".join(str(path) for path in paths),
            field=None,
            field_column="U",
            params=None,
            param_columns=None,
            max_rmse=None,
            min_r2=None,
            max_relative_l2=None,
            output=str(tmp_path / "validation.json"),
            as_json=True,
        )
        validate_payload = json.loads(capsys.readouterr().out)

        assert validate_code == 0
        assert validate_payload["validation"]["truth_shape"] == [8, 10]
        assert validate_payload["validation"]["prediction_shape"] == [8, 10]
        assert validate_payload["acceptance"]["passed"] is True
        assert "relative_l2" in validate_payload["metrics"]
        assert (tmp_path / "validation.json").exists()

        gated_code = _run_validate_twin(
            engine_path=str(tmp_path / "twin" / "engine.pkl"),
            input_path=None,
            csv_snapshots=",".join(str(path) for path in paths),
            field=None,
            field_column="U",
            params=None,
            param_columns=None,
            max_rmse=0.0,
            min_r2=None,
            max_relative_l2=None,
            output=None,
            as_json=True,
        )
        gated_payload = json.loads(capsys.readouterr().out)

        assert gated_code == 1
        assert gated_payload["status"] == "failed"
        assert gated_payload["acceptance"]["configured"] is True
        assert gated_payload["acceptance"]["checks"][0]["metric"] == "rmse"

        package_code = _run_package_twin(
            artifacts_dir=str(tmp_path / "twin"),
            include_validation=str(tmp_path / "validation.json"),
            output=str(tmp_path / "twin-delivery.zip"),
            as_json=True,
        )
        package_payload = json.loads(capsys.readouterr().out)

        assert package_code == 0
        assert package_payload["status"] == "ok"
        assert package_payload["source_integrity"]["configured"] is True
        assert package_payload["source_integrity"]["passed"] is True
        assert "engine.pkl" in package_payload["files"]
        assert "validation.json" in package_payload["files"]
        assert (tmp_path / "twin-delivery.zip").exists()

        (tmp_path / "twin" / "engine.pkl").write_bytes(b"tampered")
        tampered_code = _run_package_twin(
            artifacts_dir=str(tmp_path / "twin"),
            include_validation=str(tmp_path / "validation.json"),
            output=str(tmp_path / "tampered-delivery.zip"),
            as_json=True,
        )
        tampered_output = capsys.readouterr()

        assert tampered_code == 2
        assert "integrity mismatch" in tampered_output.err
        assert not (tmp_path / "tampered-delivery.zip").exists()

    def test_run_build_twin_reports_small_dataset(self, tmp_path, capsys) -> None:
        from naviertwin.main import _run_build_twin

        path = tmp_path / "snapshot.csv"
        path.write_text("x,U\n0,1\n1,2\n", encoding="utf-8")

        code = _run_build_twin(
            input_path=None,
            csv_snapshots=str(path),
            field=None,
            field_column="U",
            params=None,
            param_columns=None,
            outdir=str(tmp_path / "twin"),
            reducer="pod",
            n_modes=2,
            surrogate="rbf",
            validation_count=2,
            as_json=True,
        )
        output = capsys.readouterr()

        assert code == 2
        assert output.out == ""
        assert "build-twin error:" in output.err


class TestRunAutoRefine:
    def test_run_autorefine_once(self, tmp_path) -> None:
        from naviertwin.main import _run_autorefine

        src = tmp_path / "src" / "naviertwin"
        src.mkdir(parents=True)
        (src / "main.py").write_text("x=1\n", encoding="utf-8")
        (tmp_path / "ROADMAP.md").write_text(
            "- [ ] `src/naviertwin/main.py` 반영 테스트\n",
            encoding="utf-8",
        )

        code = _run_autorefine(
            interval_sec=1,
            iterations=1,
            apply=True,
            project_root=str(tmp_path),
            artifact_dir=None,
        )
        assert code == 0


class TestMain:
    def test_main_no_args_exits_zero(self, monkeypatch) -> None:
        from naviertwin.main import main

        monkeypatch.setattr(sys, "argv", ["naviertwin"])
        with pytest.raises(SystemExit) as exc:
            main()
        assert exc.value.code == 0
