"""Round 34 — CLI 서브커맨드 테스트."""

from __future__ import annotations

import subprocess
import sys


class TestCLISubcommands:
    def test_help_lists_subcommands(self) -> None:
        env_src = {"PYTHONPATH": "src"}
        import os
        env = {**os.environ, **env_src}
        result = subprocess.run(
            [sys.executable, "-m", "naviertwin.main", "--help"],
            capture_output=True, text=True, env=env,
        )
        assert result.returncode == 0
        assert "benchmark" in result.stdout
        assert "server" in result.stdout
        assert "pipeline" in result.stdout

    def test_pipeline_subcommand_runs(self) -> None:
        import os
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
