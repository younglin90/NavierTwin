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
