"""버전 문자열 정합성 테스트."""

from __future__ import annotations

from argparse import _VersionAction
from pathlib import Path

try:
    import tomllib
except ImportError:  # pragma: no cover - Python 3.10 compatibility
    import tomli as tomllib

from naviertwin import __version__
from naviertwin.main import _build_parser


def test_cli_help_contains_package_version() -> None:
    """CLI 도움말에 패키지 버전이 포함되는지 확인한다."""
    parser = _build_parser()
    help_text = parser.format_help()
    assert __version__ in help_text


def test_version_action_uses_package_version() -> None:
    """--version 액션이 패키지 버전을 사용하는지 확인한다."""
    parser = _build_parser()
    version_actions = [a for a in parser._actions if isinstance(a, _VersionAction)]
    assert version_actions, "버전 액션(--version)을 찾을 수 없습니다."
    assert any(__version__ in (a.version or "") for a in version_actions)


def test_pyproject_version_matches_package_version() -> None:
    """pyproject.toml 버전과 패키지 버전이 일치해야 한다."""
    pyproject = Path(__file__).resolve().parents[1] / "pyproject.toml"
    data = tomllib.loads(pyproject.read_text(encoding="utf-8"))
    project = data.get("project", {})
    assert project.get("version") == __version__


def test_license_metadata_is_consistent_with_license_file() -> None:
    """패키지 메타데이터의 라이선스 표기가 LICENSE 파일(MIT)과 일치해야 한다."""
    root = Path(__file__).resolve().parents[1]
    pyproject = root / "pyproject.toml"
    license_text = (root / "LICENSE").read_text(encoding="utf-8")
    data = tomllib.loads(pyproject.read_text(encoding="utf-8"))
    project = data.get("project", {})

    assert "MIT License" in license_text
    assert project.get("license") == "MIT"
    assert project.get("license-files") == ["LICENSE"]
