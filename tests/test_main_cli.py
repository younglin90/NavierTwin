"""Round 592 — main.py CLI path coverage."""

from __future__ import annotations

import json
import sys
import zipfile
from hashlib import sha256

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

    def test_parse_predict_twin_artifacts_dir_subcommand(self, tmp_path) -> None:
        from naviertwin.main import _build_parser

        p = _build_parser()
        args = p.parse_args(
            [
                "predict-twin",
                "--artifacts-dir",
                str(tmp_path / "deployed-twin"),
                "--params",
                "0.25",
                "--json",
            ]
        )
        assert args.command == "predict-twin"
        assert args.artifacts_dir.endswith("deployed-twin")
        assert args.engine is None
        assert args.params == "0.25"
        assert args.as_json is True

    def test_parse_benchmark_twin_subcommand(self, tmp_path) -> None:
        from naviertwin.main import _build_parser

        p = _build_parser()
        args = p.parse_args(
            [
                "benchmark-twin",
                "--artifacts-dir",
                str(tmp_path / "deployed-twin"),
                "--params",
                "0.25",
                "--warmup",
                "1",
                "--repeat",
                "3",
                "--max-p95-ms",
                "100",
                "--min-throughput-hz",
                "10",
                "--json",
            ]
        )
        assert args.command == "benchmark-twin"
        assert args.artifacts_dir.endswith("deployed-twin")
        assert args.params == "0.25"
        assert args.warmup == 1
        assert args.repeat == 3
        assert args.max_p95_ms == 100
        assert args.min_throughput_hz == 10
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

    def test_parse_validate_twin_artifacts_dir_subcommand(self, tmp_path) -> None:
        from naviertwin.main import _build_parser

        p = _build_parser()
        args = p.parse_args(
            [
                "validate-twin",
                "--artifacts-dir",
                str(tmp_path / "deployed-twin"),
                "--csv-snapshots",
                str(tmp_path / "snapshots"),
                "--field-column",
                "U",
                "--json",
            ]
        )
        assert args.command == "validate-twin"
        assert args.artifacts_dir.endswith("deployed-twin")
        assert args.engine is None
        assert args.csv_snapshots.endswith("snapshots")
        assert args.field_column == "U"
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

    def test_parse_verify_twin_package_subcommand(self, tmp_path) -> None:
        from naviertwin.main import _build_parser

        p = _build_parser()
        args = p.parse_args(
            [
                "verify-twin-package",
                "--package",
                str(tmp_path / "twin.zip"),
                "--extract-to",
                str(tmp_path / "deploy"),
                "--json",
            ]
        )
        assert args.command == "verify-twin-package"
        assert args.package.endswith("twin.zip")
        assert args.extract_to.endswith("deploy")
        assert args.as_json is True

    def test_parse_inspect_twin_package_subcommand(self, tmp_path) -> None:
        from naviertwin.main import _build_parser

        p = _build_parser()
        args = p.parse_args(
            [
                "inspect-twin-package",
                "--package",
                str(tmp_path / "twin.zip"),
                "--json",
            ]
        )
        assert args.command == "inspect-twin-package"
        assert args.package.endswith("twin.zip")
        assert args.as_json is True

    def test_parse_accept_twin_package_subcommand(self, tmp_path) -> None:
        from naviertwin.main import _build_parser

        p = _build_parser()
        args = p.parse_args(
            [
                "accept-twin-package",
                "--package",
                str(tmp_path / "twin.zip"),
                "--extract-to",
                str(tmp_path / "deploy"),
                "--prediction-output",
                str(tmp_path / "prediction.csv"),
                "--warmup",
                "1",
                "--repeat",
                "3",
                "--max-p95-ms",
                "100",
                "--min-throughput-hz",
                "10",
                "--output",
                str(tmp_path / "acceptance.json"),
                "--json",
            ]
        )
        assert args.command == "accept-twin-package"
        assert args.package.endswith("twin.zip")
        assert args.extract_to.endswith("deploy")
        assert args.prediction_output.endswith("prediction.csv")
        assert args.warmup == 1
        assert args.repeat == 3
        assert args.max_p95_ms == 100
        assert args.min_throughput_hz == 10
        assert args.output.endswith("acceptance.json")
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
    def test_load_build_twin_params_emits_named_contract(self, tmp_path) -> None:
        pytest.importorskip("pandas")
        from naviertwin.main import _load_build_twin_params

        path = tmp_path / "params.csv"
        path.write_text(
            "mach,aoa,note\n"
            "0.2,1.0,a\n"
            "0.4,2.0,b\n"
            "0.6,3.0,c\n",
            encoding="utf-8",
        )

        values, contract = _load_build_twin_params(
            params_path=str(path),
            param_columns="mach,aoa",
            n_snapshots=3,
        )

        assert values.shape == (3, 2)
        assert contract["dim"] == 2
        assert contract["names"] == ["mach", "aoa"]
        assert contract["source"].endswith("params.csv")
        assert contract["ranges"] == [
            {"name": "mach", "observed_min": 0.2, "observed_max": 0.6},
            {"name": "aoa", "observed_min": 1.0, "observed_max": 3.0},
        ]

    def test_parameter_contract_check_allows_legacy_missing_contract(self) -> None:
        from naviertwin.main import _check_twin_parameter_contract

        check = _check_twin_parameter_contract(np.array([0.25]), None)

        assert check["available"] is False
        assert check["passed"] is True

    def test_run_build_twin_from_csv_snapshots(self, tmp_path, capsys) -> None:
        pytest.importorskip("h5py")
        pytest.importorskip("pandas")
        pytest.importorskip("sklearn")
        from naviertwin.main import (
            _run_accept_twin_package,
            _run_benchmark_twin,
            _run_build_twin,
            _run_inspect_twin_package,
            _run_package_twin,
            _run_predict_twin,
            _run_validate_twin,
            _run_verify_twin_package,
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
        contract = payload["training"]["parameter_contract"]
        assert contract["schema"] == "naviertwin-parameter-contract-v1"
        assert contract["dim"] == 1
        assert contract["names"] == ["normalized_index"]
        assert contract["ranges"][0]["observed_min"] == 0.0
        assert contract["ranges"][0]["observed_max"] == 1.0

        from naviertwin.utils.hashing import hash_file

        manifest = json.loads((tmp_path / "twin" / "manifest.json").read_text(encoding="utf-8"))
        integrity = manifest["extra"]["artifact_integrity"]
        assert set(integrity) == {"metrics", "checkpoint", "engine", "report"}
        assert integrity["engine"]["sha256"] == hash_file(tmp_path / "twin" / "engine.pkl")
        assert integrity["metrics"]["bytes"] > 0
        assert manifest["extra"]["parameter_contract"] == contract

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
        assert predict_payload["parameter_contract"]["dim"] == 1
        assert predict_payload["parameter_check"]["available"] is True
        assert predict_payload["parameter_check"]["input_dim"] == 1
        assert (tmp_path / "prediction.csv").exists()

        mismatch_predict_code = _run_predict_twin(
            engine_path=str(tmp_path / "twin" / "engine.pkl"),
            params="0.25,0.5",
            params_csv=None,
            param_columns=None,
            output=None,
            as_json=True,
        )
        mismatch_output = capsys.readouterr()

        assert mismatch_predict_code == 2
        assert mismatch_output.out == ""
        assert "parameter dimension mismatch" in mismatch_output.err

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
        assert package_payload["generated_entries"] == [
            "README.txt",
            "delivery.json",
            "sample_params.csv",
        ]
        assert "README.txt" not in package_payload["files"]
        assert "delivery.json" not in package_payload["files"]
        manifest_names = {entry["name"] for entry in package_payload["manifest_entries"]}
        assert {"README.txt", "delivery.json", "sample_params.csv"} <= manifest_names
        assert (tmp_path / "twin-delivery.zip").exists()

        with zipfile.ZipFile(tmp_path / "twin-delivery.zip") as archive:
            readme = archive.read("README.txt").decode("utf-8")
            delivery = json.loads(archive.read("delivery.json").decode("utf-8"))
            sample_params = archive.read("sample_params.csv").decode("utf-8")
        assert "verify-twin-package" in readme
        assert "--max-p95-ms" in readme
        assert "Expected input parameters" in readme
        assert delivery["format"] == "NavierTwin delivery package"
        assert delivery["commands"]["predict"].startswith("naviertwin predict-twin")
        assert "--params-csv <extracted-dir>/sample_params.csv" in delivery["commands"]["predict"]
        assert "--param-columns normalized_index" in delivery["commands"]["benchmark"]
        assert "--min-throughput-hz" in delivery["commands"]["benchmark"]
        assert sample_params == "normalized_index\n0.5\n"
        assert delivery["parameter_contract"] == contract

        inspect_code = _run_inspect_twin_package(
            package_path=str(tmp_path / "twin-delivery.zip"),
            as_json=True,
        )
        inspect_payload = json.loads(capsys.readouterr().out)

        assert inspect_code == 0
        assert inspect_payload["status"] == "ok"
        assert inspect_payload["format"] == "NavierTwin delivery package"
        assert inspect_payload["delivery_metadata_present"] is True
        assert inspect_payload["validation_included"] is True
        assert inspect_payload["readme_present"] is True
        assert inspect_payload["verification"]["status"] == "ok"
        assert "rmse" in inspect_payload["metrics"]
        assert inspect_payload["parameter_contract"] == contract

        verify_code = _run_verify_twin_package(
            package_path=str(tmp_path / "twin-delivery.zip"),
            extract_to=str(tmp_path / "deployed-twin"),
            as_json=True,
        )
        verify_payload = json.loads(capsys.readouterr().out)

        assert verify_code == 0
        assert verify_payload["status"] == "ok"
        assert verify_payload["manifest_entry_count"] >= 7
        assert verify_payload["extracted_to"].endswith("deployed-twin")
        assert "engine.pkl" in verify_payload["extracted_entries"]
        assert not verify_payload["errors"]
        assert (tmp_path / "deployed-twin" / "engine.pkl").exists()
        assert (tmp_path / "deployed-twin" / "README.txt").exists()

        accept_code = _run_accept_twin_package(
            package_path=str(tmp_path / "twin-delivery.zip"),
            extract_to=str(tmp_path / "accepted-twin"),
            prediction_output=str(tmp_path / "accept-prediction.csv"),
            warmup=0,
            repeat=2,
            max_mean_ms=None,
            max_p50_ms=None,
            max_p95_ms=100000,
            max_p99_ms=None,
            min_throughput_hz=0.0001,
            skip_benchmark=False,
            output=str(tmp_path / "acceptance.json"),
            as_json=True,
        )
        accept_payload = json.loads(capsys.readouterr().out)

        assert accept_code == 0
        assert accept_payload["status"] == "ok"
        assert accept_payload["acceptance"]["package"] is True
        assert accept_payload["acceptance"]["prediction"] is True
        assert accept_payload["acceptance"]["benchmark"] is True
        assert accept_payload["verification"]["status"] == "ok"
        assert accept_payload["inspection"]["delivery_metadata_present"] is True
        assert accept_payload["parameter_input"]["source"] == "sample_params.csv"
        assert accept_payload["prediction"]["prediction_shape"] == [8, 1]
        assert accept_payload["prediction"]["parameter_check"]["expected_dim"] == 1
        assert accept_payload["benchmark"]["repeat"] == 2
        assert len(accept_payload["benchmark"]["samples_ms"]) == 2
        assert accept_payload["benchmark"]["acceptance"]["configured"] is True
        assert (tmp_path / "accepted-twin" / "engine.pkl").exists()
        assert (tmp_path / "accept-prediction.csv").exists()
        assert (tmp_path / "acceptance.json").exists()

        gated_accept_code = _run_accept_twin_package(
            package_path=str(tmp_path / "twin-delivery.zip"),
            extract_to=str(tmp_path / "accepted-twin-gated"),
            prediction_output=None,
            warmup=0,
            repeat=1,
            max_mean_ms=0.0,
            max_p50_ms=None,
            max_p95_ms=None,
            max_p99_ms=None,
            min_throughput_hz=None,
            skip_benchmark=False,
            output=None,
            as_json=True,
        )
        gated_accept_payload = json.loads(capsys.readouterr().out)

        assert gated_accept_code == 1
        assert gated_accept_payload["status"] == "failed"
        assert gated_accept_payload["acceptance"]["package"] is True
        assert gated_accept_payload["acceptance"]["prediction"] is True
        assert gated_accept_payload["acceptance"]["benchmark"] is False
        assert gated_accept_payload["benchmark"]["acceptance"]["checks"][0]["metric"] == "latency_ms.mean"

        deployed_predict_code = _run_predict_twin(
            engine_path=None,
            artifacts_dir=str(tmp_path / "deployed-twin"),
            params="0.25",
            params_csv=None,
            param_columns=None,
            output=str(tmp_path / "deployed-prediction.csv"),
            as_json=True,
        )
        deployed_predict_payload = json.loads(capsys.readouterr().out)

        assert deployed_predict_code == 0
        assert deployed_predict_payload["artifacts_dir"].endswith("deployed-twin")
        assert deployed_predict_payload["engine"].endswith("engine.pkl")
        assert (tmp_path / "deployed-prediction.csv").exists()

        benchmark_code = _run_benchmark_twin(
            engine_path=None,
            artifacts_dir=str(tmp_path / "deployed-twin"),
            params="0.25",
            params_csv=None,
            param_columns=None,
            warmup=1,
            repeat=3,
            output=str(tmp_path / "latency.json"),
            as_json=True,
        )
        benchmark_payload = json.loads(capsys.readouterr().out)

        assert benchmark_code == 0
        assert benchmark_payload["artifacts_dir"].endswith("deployed-twin")
        assert benchmark_payload["repeat"] == 3
        assert len(benchmark_payload["samples_ms"]) == 3
        assert benchmark_payload["latency_ms"]["p95"] >= benchmark_payload["latency_ms"]["min"]
        assert benchmark_payload["parameter_check"]["available"] is True
        assert benchmark_payload["parameter_check"]["expected_dim"] == 1
        assert benchmark_payload["acceptance"]["passed"] is True
        assert benchmark_payload["acceptance"]["configured"] is False
        assert (tmp_path / "latency.json").exists()

        mismatch_benchmark_code = _run_benchmark_twin(
            engine_path=None,
            artifacts_dir=str(tmp_path / "deployed-twin"),
            params="0.25,0.5",
            params_csv=None,
            param_columns=None,
            warmup=0,
            repeat=1,
            output=None,
            as_json=True,
        )
        mismatch_benchmark_output = capsys.readouterr()

        assert mismatch_benchmark_code == 2
        assert mismatch_benchmark_output.out == ""
        assert "parameter dimension mismatch" in mismatch_benchmark_output.err

        gated_benchmark_code = _run_benchmark_twin(
            engine_path=None,
            artifacts_dir=str(tmp_path / "deployed-twin"),
            params="0.25",
            params_csv=None,
            param_columns=None,
            warmup=0,
            repeat=1,
            max_mean_ms=0.0,
            max_p50_ms=None,
            max_p95_ms=None,
            max_p99_ms=None,
            min_throughput_hz=None,
            output=None,
            as_json=True,
        )
        gated_benchmark_payload = json.loads(capsys.readouterr().out)

        assert gated_benchmark_code == 1
        assert gated_benchmark_payload["status"] == "failed"
        assert gated_benchmark_payload["acceptance"]["configured"] is True
        assert gated_benchmark_payload["acceptance"]["checks"][0]["metric"] == "latency_ms.mean"

        deployed_validate_code = _run_validate_twin(
            engine_path=None,
            artifacts_dir=str(tmp_path / "deployed-twin"),
            input_path=None,
            csv_snapshots=",".join(str(path) for path in paths),
            field=None,
            field_column="U",
            params=None,
            param_columns=None,
            max_rmse=None,
            min_r2=None,
            max_relative_l2=None,
            output=str(tmp_path / "deployed-validation.json"),
            as_json=True,
        )
        deployed_validate_payload = json.loads(capsys.readouterr().out)

        assert deployed_validate_code == 0
        assert deployed_validate_payload["artifacts_dir"].endswith("deployed-twin")
        assert deployed_validate_payload["engine"].endswith("engine.pkl")
        assert (tmp_path / "deployed-validation.json").exists()

        repeat_extract_code = _run_verify_twin_package(
            package_path=str(tmp_path / "twin-delivery.zip"),
            extract_to=str(tmp_path / "deployed-twin"),
            as_json=True,
        )
        repeat_extract_output = capsys.readouterr()

        assert repeat_extract_code == 2
        assert repeat_extract_output.out == ""
        assert "extract target must be empty" in repeat_extract_output.err

        bad_zip = tmp_path / "bad-delivery.zip"
        with zipfile.ZipFile(bad_zip, "w") as archive:
            archive.writestr("engine.pkl", b"tampered")
            archive.writestr("manifest.json", b"{}")
            archive.writestr(
                "MANIFEST.json",
                json.dumps(
                    [
                        {"name": "engine.pkl", "bytes": 999, "sha256": "0" * 64},
                        {
                            "name": "manifest.json",
                            "bytes": 2,
                            "sha256": sha256(b"{}").hexdigest(),
                        },
                    ]
                ),
            )
        bad_code = _run_verify_twin_package(
            package_path=str(bad_zip),
            extract_to=None,
            as_json=True,
        )
        bad_payload = json.loads(capsys.readouterr().out)

        assert bad_code == 1
        assert bad_payload["status"] == "failed"
        assert any("integrity mismatch" in error for error in bad_payload["errors"])

        duplicate_zip = tmp_path / "duplicate-delivery.zip"
        engine_data = b"engine"
        manifest_data = b"{}"
        with zipfile.ZipFile(duplicate_zip, "w") as archive:
            archive.writestr("engine.pkl", b"shadow")
            with pytest.warns(UserWarning, match="Duplicate name"):
                archive.writestr("engine.pkl", engine_data)
            archive.writestr("manifest.json", manifest_data)
            archive.writestr(
                "MANIFEST.json",
                json.dumps(
                    [
                        {
                            "name": "engine.pkl",
                            "bytes": len(engine_data),
                            "sha256": sha256(engine_data).hexdigest(),
                        },
                        {
                            "name": "manifest.json",
                            "bytes": len(manifest_data),
                            "sha256": sha256(manifest_data).hexdigest(),
                        },
                    ]
                ),
            )
        duplicate_code = _run_verify_twin_package(
            package_path=str(duplicate_zip),
            extract_to=None,
            as_json=True,
        )
        duplicate_payload = json.loads(capsys.readouterr().out)

        assert duplicate_code == 1
        assert duplicate_payload["status"] == "failed"
        assert duplicate_payload["duplicate_archive_entries"] == ["engine.pkl"]
        assert "duplicate archive entry: engine.pkl" in duplicate_payload["errors"]

        unsafe_zip = tmp_path / "unsafe-delivery.zip"
        unsafe_data = b"escape"
        with zipfile.ZipFile(unsafe_zip, "w") as archive:
            archive.writestr("engine.pkl", engine_data)
            archive.writestr("manifest.json", manifest_data)
            archive.writestr("../evil.txt", unsafe_data)
            archive.writestr(
                "MANIFEST.json",
                json.dumps(
                    [
                        {
                            "name": "engine.pkl",
                            "bytes": len(engine_data),
                            "sha256": sha256(engine_data).hexdigest(),
                        },
                        {
                            "name": "manifest.json",
                            "bytes": len(manifest_data),
                            "sha256": sha256(manifest_data).hexdigest(),
                        },
                        {
                            "name": "../evil.txt",
                            "bytes": len(unsafe_data),
                            "sha256": sha256(unsafe_data).hexdigest(),
                        },
                    ]
                ),
            )
        unsafe_code = _run_verify_twin_package(
            package_path=str(unsafe_zip),
            extract_to=str(tmp_path / "unsafe-out"),
            as_json=True,
        )
        unsafe_payload = json.loads(capsys.readouterr().out)

        assert unsafe_code == 1
        assert unsafe_payload["status"] == "failed"
        assert "unsafe archive entry: ../evil.txt" in unsafe_payload["errors"]
        assert unsafe_payload["extracted_entries"] == []
        assert not (tmp_path / "evil.txt").exists()

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

    def test_package_twin_uses_multidimensional_contract_commands(
        self, tmp_path, capsys
    ) -> None:
        pytest.importorskip("h5py")
        pytest.importorskip("pandas")
        pytest.importorskip("sklearn")
        from naviertwin.main import (
            _run_build_twin,
            _run_inspect_twin_package,
            _run_package_twin,
        )

        paths = []
        param_rows = ["mach,aoa"]
        for step in range(8):
            path = tmp_path / f"snapshot_{step:03d}.csv"
            rows = ["x,U"]
            for index in range(6):
                rows.append(f"{index},{step * 0.15 + index * 0.02}")
            path.write_text("\n".join(rows) + "\n", encoding="utf-8")
            paths.append(path)
            param_rows.append(f"{0.2 + step * 0.1},{1.0 + step * 0.5}")
        params_path = tmp_path / "params.csv"
        params_path.write_text("\n".join(param_rows) + "\n", encoding="utf-8")

        build_code = _run_build_twin(
            input_path=None,
            csv_snapshots=",".join(str(path) for path in paths),
            field=None,
            field_column="U",
            params=str(params_path),
            param_columns="mach,aoa",
            outdir=str(tmp_path / "twin2d"),
            reducer="pod",
            n_modes=2,
            surrogate="rbf",
            validation_count=2,
            as_json=True,
        )
        build_payload = json.loads(capsys.readouterr().out)

        assert build_code == 0
        contract = build_payload["training"]["parameter_contract"]
        assert contract["dim"] == 2
        assert contract["names"] == ["mach", "aoa"]

        package_code = _run_package_twin(
            artifacts_dir=str(tmp_path / "twin2d"),
            include_validation=None,
            output=str(tmp_path / "twin2d-delivery.zip"),
            as_json=True,
        )
        assert package_code == 0
        capsys.readouterr()

        with zipfile.ZipFile(tmp_path / "twin2d-delivery.zip") as archive:
            readme = archive.read("README.txt").decode("utf-8")
            delivery = json.loads(archive.read("delivery.json").decode("utf-8"))
            original_entries = {
                name: archive.read(name)
                for name in archive.namelist()
                if name != "MANIFEST.json"
            }

        sample_params = original_entries["sample_params.csv"].decode("utf-8")

        assert "--params-csv <extracted-dir>/sample_params.csv" in delivery["commands"]["predict"]
        assert "--param-columns mach,aoa" in delivery["commands"]["benchmark"]
        assert "--params-csv <extracted-dir>/sample_params.csv" in readme
        assert sample_params == "mach,aoa\n0.55,2.75\n"
        assert delivery["parameter_contract"] == contract

        inspect_code = _run_inspect_twin_package(
            package_path=str(tmp_path / "twin2d-delivery.zip"),
            as_json=True,
        )
        inspect_payload = json.loads(capsys.readouterr().out)

        assert inspect_code == 0
        assert inspect_payload["parameter_contract"] == contract

        tampered_delivery = dict(delivery)
        tampered_delivery["parameter_contract"] = {
            **contract,
            "dim": 1,
            "names": ["wrong"],
        }
        original_entries["delivery.json"] = (
            json.dumps(tampered_delivery, ensure_ascii=False, sort_keys=True, indent=2)
            + "\n"
        ).encode("utf-8")
        tampered_zip = tmp_path / "contract-mismatch.zip"
        manifest_entries = [
            {
                "name": name,
                "bytes": len(data),
                "sha256": sha256(data).hexdigest(),
            }
            for name, data in sorted(original_entries.items())
        ]
        with zipfile.ZipFile(tampered_zip, "w") as archive:
            for name, data in original_entries.items():
                archive.writestr(name, data)
            archive.writestr("MANIFEST.json", json.dumps(manifest_entries))

        mismatch_code = _run_inspect_twin_package(
            package_path=str(tampered_zip),
            as_json=True,
        )
        mismatch_payload = json.loads(capsys.readouterr().out)

        assert mismatch_code == 1
        assert mismatch_payload["status"] == "failed"
        assert "delivery.json parameter_contract differs from manifest.json" in mismatch_payload["errors"]

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
