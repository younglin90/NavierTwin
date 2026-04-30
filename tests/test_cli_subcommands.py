"""Round 34 — CLI 서브커맨드 테스트."""

from __future__ import annotations

import json
import os
import subprocess
import sys

EXPECTED_SUBCOMMANDS = [
    "benchmark",
    "server",
    "pipeline",
    "pipeline-demo",
    "model-sweep",
    "build-twin",
    "predict-twin",
    "validate-twin",
    "package-twin",
    "verify-twin-package",
    "preflight",
    "support-bundle",
    "autorefine",
    "update-check",
    "doctor",
]


class TestCLISubcommands:
    def test_help_lists_subcommands(self) -> None:
        env_src = {"PYTHONPATH": "src"}
        env = {**os.environ, **env_src}
        result = subprocess.run(
            [sys.executable, "-m", "naviertwin.main", "--help"],
            capture_output=True, text=True, env=env,
        )
        assert result.returncode == 0
        for command in EXPECTED_SUBCOMMANDS:
            assert command in result.stdout

    def test_server_subcommand_help_is_copy_pasteable(self) -> None:
        env = {**os.environ, "PYTHONPATH": "src"}
        result = subprocess.run(
            [sys.executable, "-m", "naviertwin.main", "server", "--help"],
            capture_output=True,
            text=True,
            env=env,
        )
        assert result.returncode == 0
        assert "usage: naviertwin server" in result.stdout
        assert "--host" in result.stdout
        assert "--port" in result.stdout

    def test_pipeline_subcommand_runs(self) -> None:
        env = {**os.environ, "PYTHONPATH": "src"}
        result = subprocess.run(
            [
                sys.executable, "-m", "naviertwin.main",
                "pipeline", "--reducer", "pod", "--n-modes", "3",
                "--surrogate", "rbf",
            ],
            capture_output=True, text=True, env=env,
        )
        assert result.returncode == 0
        assert "파이프라인 완료" in result.stdout or "rmse" in result.stdout

    def test_model_sweep_subcommand_runs_json(self) -> None:
        env = {**os.environ, "PYTHONPATH": "src"}
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "naviertwin.main",
                "model-sweep",
                "--reducers",
                "pod",
                "--n-modes",
                "2,3",
                "--surrogates",
                "rbf",
                "--samples",
                "14",
                "--features",
                "18",
                "--json",
            ],
            capture_output=True, text=True, env=env,
        )
        assert result.returncode == 0, result.stderr

        payload = json.loads(result.stdout)
        assert payload["status"] == "ok"
        assert payload["configs"] == 2
        assert len(payload["rows"]) == 2
        assert payload["best"]["reducer_kind"] == "pod"

    def test_build_twin_subcommand_runs_json(self, tmp_path) -> None:
        paths = []
        for step in range(10):
            path = tmp_path / f"snapshot_{step:03d}.csv"
            rows = ["x,U"]
            for index in range(8):
                rows.append(f"{index},{step * 0.2 + index * 0.01}")
            path.write_text("\n".join(rows) + "\n", encoding="utf-8")
            paths.append(path)

        outdir = tmp_path / "twin"
        env = {**os.environ, "PYTHONPATH": "src"}
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "naviertwin.main",
                "build-twin",
                "--csv-snapshots",
                ",".join(str(path) for path in paths),
                "--field-column",
                "U",
                "--outdir",
                str(outdir),
                "--n-modes",
                "2",
                "--surrogate",
                "rbf",
                "--validation-count",
                "2",
                "--json",
            ],
            capture_output=True, text=True, env=env,
        )
        assert result.returncode == 0, result.stderr

        payload = json.loads(result.stdout)
        assert payload["status"] == "ok"
        assert payload["training"]["n_snapshots"] == 10
        assert payload["training"]["validation_count"] == 2
        assert (outdir / "pipeline.h5").exists()
        assert (outdir / "engine.pkl").exists()
        assert (outdir / "manifest.json").exists()

        predict_result = subprocess.run(
            [
                sys.executable,
                "-m",
                "naviertwin.main",
                "predict-twin",
                "--engine",
                str(outdir / "engine.pkl"),
                "--params",
                "0.25",
                "--output",
                str(tmp_path / "prediction.csv"),
                "--json",
            ],
            capture_output=True,
            text=True,
            env=env,
        )
        assert predict_result.returncode == 0, predict_result.stderr

        prediction_payload = json.loads(predict_result.stdout)
        assert prediction_payload["status"] == "ok"
        assert prediction_payload["input_shape"] == [1]
        assert prediction_payload["prediction_shape"] == [8]
        assert (tmp_path / "prediction.csv").exists()

        validate_result = subprocess.run(
            [
                sys.executable,
                "-m",
                "naviertwin.main",
                "validate-twin",
                "--engine",
                str(outdir / "engine.pkl"),
                "--csv-snapshots",
                ",".join(str(path) for path in paths),
                "--field-column",
                "U",
                "--output",
                str(tmp_path / "validation.json"),
                "--json",
            ],
            capture_output=True,
            text=True,
            env=env,
        )
        assert validate_result.returncode == 0, validate_result.stderr

        validation_payload = json.loads(validate_result.stdout)
        assert validation_payload["status"] == "ok"
        assert validation_payload["validation"]["truth_shape"] == [8, 10]
        assert validation_payload["validation"]["prediction_shape"] == [8, 10]
        assert "rmse" in validation_payload["metrics"]
        assert (tmp_path / "validation.json").exists()

        package_result = subprocess.run(
            [
                sys.executable,
                "-m",
                "naviertwin.main",
                "package-twin",
                "--artifacts-dir",
                str(outdir),
                "--include-validation",
                str(tmp_path / "validation.json"),
                "--output",
                str(tmp_path / "twin-delivery.zip"),
                "--json",
            ],
            capture_output=True,
            text=True,
            env=env,
        )
        assert package_result.returncode == 0, package_result.stderr

        package_payload = json.loads(package_result.stdout)
        assert package_payload["status"] == "ok"
        assert "engine.pkl" in package_payload["files"]
        assert "validation.json" in package_payload["files"]
        assert (tmp_path / "twin-delivery.zip").exists()

        verify_package_result = subprocess.run(
            [
                sys.executable,
                "-m",
                "naviertwin.main",
                "verify-twin-package",
                "--package",
                str(tmp_path / "twin-delivery.zip"),
                "--json",
            ],
            capture_output=True,
            text=True,
            env=env,
        )
        assert verify_package_result.returncode == 0, verify_package_result.stderr

        verify_payload = json.loads(verify_package_result.stdout)
        assert verify_payload["status"] == "ok"
        assert verify_payload["manifest_entry_count"] >= 5
