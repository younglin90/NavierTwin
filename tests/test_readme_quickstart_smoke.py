"""README quickstart smoke commands must stay copy-pasteable."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
README = ROOT / "README.md"


def _env() -> dict[str, str]:
    env = {
        **os.environ,
        "PYTHONPATH": str(ROOT / "src"),
        "QT_QPA_PLATFORM": "offscreen",
        "MPLCONFIGDIR": "/tmp/mpl",
        # 호스트 로케일이 C/POSIX(ASCII)인 CI/샌드박스에서도 자식 프로세스가
        # 항상 UTF-8 로 stdout/stderr 를 쓰게 강제한다 — 안 그러면 한국어
        # 메시지가 섞인 출력을 부모가 디코드할 때 UnicodeDecodeError 가 난다.
        "PYTHONIOENCODING": "utf-8",
        "PYTHONUTF8": "1",
    }
    return env


def test_readme_documents_minimal_quickstart_smoke() -> None:
    """README must expose a deterministic, local install check."""
    text = README.read_text(encoding="utf-8")

    assert "# 최소 quickstart smoke" in text
    assert "naviertwin --version" in text
    assert "naviertwin preflight tests/fixtures/tiny_square.su2 --json" in text


def test_readme_minimal_quickstart_commands_execute() -> None:
    """Run the README quickstart smoke without console-script install assumptions."""
    version = subprocess.run(
        [sys.executable, "-m", "naviertwin.main", "--version"],
        cwd=ROOT,
        env=_env(),
        capture_output=True,
        text=True,
        encoding="utf-8",
        check=False,
    )
    assert version.returncode == 0
    assert version.stdout.startswith("naviertwin ")
    assert "Traceback" not in version.stderr

    preflight = subprocess.run(
        [
            sys.executable,
            "-m",
            "naviertwin.main",
            "preflight",
            "tests/fixtures/tiny_square.su2",
            "--json",
        ],
        cwd=ROOT,
        env=_env(),
        capture_output=True,
        text=True,
        encoding="utf-8",
        check=False,
    )
    payload = json.loads(preflight.stdout)

    assert preflight.returncode == 0
    assert payload["status"] == "ok"
    assert payload["readiness_score"] == 100
    assert "Traceback" not in preflight.stderr
