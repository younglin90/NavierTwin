"""패키징 메타데이터(옵셔널 의존성) 회귀 테스트."""

from __future__ import annotations

from pathlib import Path

try:
    import tomllib
except ImportError:  # pragma: no cover - Python 3.10 compatibility
    import tomli as tomllib

from scripts.release_versions import example_latest_release_version, project_version


def _project_optional_deps() -> dict[str, list[str]]:
    root = Path(__file__).resolve().parents[1]
    pyproject = root / "pyproject.toml"
    data = tomllib.loads(pyproject.read_text(encoding="utf-8"))
    project = data.get("project", {})
    return project.get("optional-dependencies", {})


def _project_metadata() -> dict[str, object]:
    root = Path(__file__).resolve().parents[1]
    pyproject = root / "pyproject.toml"
    data = tomllib.loads(pyproject.read_text(encoding="utf-8"))
    return data.get("project", {})


def test_full_extra_includes_core_runtime_optional_stack() -> None:
    """full extra가 주요 선택 런타임 스택을 포함해야 한다."""
    optional_deps = _project_optional_deps()
    full = optional_deps.get("full", [])

    required_prefixes = [
        "naviertwin[core]",
        "pandas",
        "pyarrow",
        "zarr",
        "xarray",
        "netCDF4",
        "PyWavelets",
        "PyYAML",
        "botorch",
        "pymor",
        "nlopt",
    ]

    for prefix in required_prefixes:
        assert any(dep.startswith(prefix) for dep in full), (
            f"full extra에 '{prefix}' 의존성이 없습니다."
        )


def test_dev_extra_includes_release_validation_tooling() -> None:
    """dev extra가 릴리스 검증 도구를 포함해야 한다."""
    optional_deps = _project_optional_deps()
    dev = optional_deps.get("dev", [])

    required_prefixes = [
        "pytest",
        "pytest-qt",
        "pytest-cov",
        "build",
        "twine",
        "ruff",
        "isort",
        "mypy",
    ]

    for prefix in required_prefixes:
        assert any(dep.startswith(prefix) for dep in dev), (
            f"dev extra에 '{prefix}' 의존성이 없습니다."
        )


def test_dev_extra_includes_python310_toml_fallback() -> None:
    """Python 3.10 개발/테스트 환경은 tomllib 대체재가 필요하다."""
    optional_deps = _project_optional_deps()
    dev = optional_deps.get("dev", [])

    assert any(
        dep.startswith("tomli>=2.0") and "python_version < '3.11'" in dep
        for dep in dev
    )


def test_release_classifier_is_not_alpha() -> None:
    """고객 검증용 릴리스 메타데이터는 alpha 상태로 배포하지 않는다."""
    metadata = _project_metadata()
    classifiers = metadata.get("classifiers", [])

    assert "Development Status :: 3 - Alpha" not in classifiers
    assert "License :: OSI Approved :: MIT License" not in classifiers
    assert metadata.get("license") == "MIT"
    assert metadata.get("license-files") == ["LICENSE"]
    assert any(
        classifier
        in {
            "Development Status :: 4 - Beta",
            "Development Status :: 5 - Production/Stable",
        }
        for classifier in classifiers
    )


def test_release_docs_do_not_conflict_with_packaging_policy() -> None:
    """공개 문서가 MIT core/optional extra 배포 정책과 충돌하지 않아야 한다."""
    root = Path(__file__).resolve().parents[1]

    readme = (root / "README.md").read_text(encoding="utf-8")
    agents = (root / "AGENTS.md").read_text(encoding="utf-8")
    plan = (root / "PLAN.md").read_text(encoding="utf-8")
    spec = (root / "SPEC.md").read_text(encoding="utf-8")

    assert "naviertwin --gui" in readme
    assert "python3 main.py --gui" not in readme
    assert "naviertwin server --host 0.0.0.0 --port 8000" in readme
    assert "python scripts/release_smoke.py" in readme
    assert "python scripts/wheel_smoke.py --install-smoke" in readme
    assert "python scripts/sdist_smoke.py --install-smoke" in readme
    assert "pytest --collect-only -q" in readme
    assert "NAVIER_TWIN_RUN_PYMOR=1" in readme
    assert "`POST /reduce`" in readme
    assert "`POST /simulate/lbm_cavity`" in readme
    assert "비상업용" not in agents
    assert "v4.2.0 + 17 rounds" not in agents
    assert "GPL-3.0 오픈소스" not in plan
    assert "비상업용" not in spec
    assert "GPL-3.0 비상업용" not in spec


def test_readme_documents_shipped_cli_surface() -> None:
    """README quickstart must show the shipped non-GUI command surface."""
    root = Path(__file__).resolve().parents[1]
    readme = (root / "README.md").read_text(encoding="utf-8")

    expected_commands = [
        "naviertwin update-check --metadata examples/release-metadata.example.json",
        "naviertwin benchmark --kind burgers",
        "naviertwin server --host 0.0.0.0 --port 8000",
        "naviertwin autorefine --iterations 1 --dry-run",
        "naviertwin doctor --json",
        "naviertwin preflight tests/fixtures/tiny_square.su2 --json --output",
        "naviertwin support-bundle --outdir",
        "python scripts/license_report.py --json --output",
        "naviertwin pipeline-demo --outdir",
        "naviertwin model-sweep --reducers pod",
    ]

    for command in expected_commands:
        assert command in readme


def test_gui_runtime_assets_are_included_in_package_data() -> None:
    """wheel/sdist가 GUI 테마와 i18n JSON 런타임 자산을 포함해야 한다."""
    root = Path(__file__).resolve().parents[1]
    pyproject = root / "pyproject.toml"
    manifest = root / "MANIFEST.in"

    data = tomllib.loads(pyproject.read_text(encoding="utf-8"))
    setuptools_cfg = data.get("tool", {}).get("setuptools", {})
    package_data = setuptools_cfg.get("package-data", {})
    naviertwin_data = package_data.get("naviertwin", [])
    manifest_text = manifest.read_text(encoding="utf-8")

    assert setuptools_cfg.get("include-package-data") is True
    assert "gui/styles/*.qss" in naviertwin_data
    assert "gui/styles/i18n/*.json" in naviertwin_data
    assert "recursive-include src/naviertwin/gui/styles *.qss *.json" in manifest_text
    assert (
        "include scripts/release_smoke.py scripts/wheel_smoke.py scripts/sdist_smoke.py "
        "scripts/license_report.py scripts/release_versions.py"
    ) in manifest_text


def test_smoke_scripts_do_not_hardcode_update_check_versions() -> None:
    """wheel/sdist smoke는 update-check 버전 리터럴을 하드코딩하지 않아야 한다."""
    root = Path(__file__).resolve().parents[1]
    wheel_smoke = (root / "scripts" / "wheel_smoke.py").read_text(encoding="utf-8")
    sdist_smoke = (root / "scripts" / "sdist_smoke.py").read_text(encoding="utf-8")
    current_version = project_version()
    latest_example = example_latest_release_version()

    for script_text in (wheel_smoke, sdist_smoke):
        assert "\"--current-version\",\n        current_version," in script_text
        assert "expected_latest_version = example_latest_release_version()" in script_text
        assert current_version not in script_text
        assert latest_example not in script_text


def test_smoke_scripts_enforce_active_metadata_contract_checks() -> None:
    """wheel/sdist smoke scripts must actively parse and validate packaging metadata."""
    root = Path(__file__).resolve().parents[1]
    wheel_smoke = (root / "scripts" / "wheel_smoke.py").read_text(encoding="utf-8")
    sdist_smoke = (root / "scripts" / "sdist_smoke.py").read_text(encoding="utf-8")

    for script_text in (wheel_smoke, sdist_smoke):
        assert "entry_points.txt" in script_text
        assert "Requires-Python" in script_text
        assert "Provides-Extra" in script_text
        assert "Requires-Dist" in script_text
        assert "Parser().parsestr" in script_text
        assert "configparser.ConfigParser()" in script_text
        assert "naviertwin.main:main" in script_text
        assert "tomli>=2.0; python_version < \\\"3.11\\\" and extra == \\\"dev\\\"" in script_text
        assert '{"core", "full", "dev"}' in script_text

    assert "src/naviertwin.egg-info/entry_points.txt" in sdist_smoke
    assert "src/naviertwin.egg-info/requires.txt" in sdist_smoke
    assert "[dev:python_version < \\\"3.11\\\"]" in sdist_smoke


def test_installed_artifact_smokes_enforce_repo_isolation() -> None:
    """Installed artifact smoke must not pass because repo cwd/PYTHONPATH leaks in."""
    root = Path(__file__).resolve().parents[1]
    wheel_smoke = (root / "scripts" / "wheel_smoke.py").read_text(encoding="utf-8")
    sdist_smoke = (root / "scripts" / "sdist_smoke.py").read_text(encoding="utf-8")

    for script_text in (wheel_smoke, sdist_smoke):
        assert "_enter_installed_runtime" in script_text
        assert "runtime-cwd" in script_text
        assert 'os.environ.pop("PYTHONPATH", None)' in script_text
        assert 'os.environ["PYTHONNOUSERSITE"] = "1"' in script_text
        assert "--force-reinstall" in script_text
        assert 'install_env.pop("PYTHONPATH", None)' in script_text
        assert "allowed_statuses={\"ok\", \"warn\", \"error\"}" in script_text
        assert "importlib.import_module(name).__file__" in script_text
        assert "['naviertwin', 'naviertwin.core', 'naviertwin.utils']" in script_text
        assert "installed import for {module_name} resolved to repo checkout" in script_text
        assert "installed import for {module_name} did not resolve inside venv" in script_text

    assert "system_site_packages=True" not in sdist_smoke
    assert "_build_wheel_from_sdist" in sdist_smoke
    assert "wheel-from-sdist" in sdist_smoke


def test_installed_artifact_smokes_enforce_support_bundle_privacy() -> None:
    """wheel/sdist install smoke must check redaction in installed support bundles."""
    root = Path(__file__).resolve().parents[1]
    wheel_smoke = (root / "scripts" / "wheel_smoke.py").read_text(encoding="utf-8")
    sdist_smoke = (root / "scripts" / "sdist_smoke.py").read_text(encoding="utf-8")

    for script_text in (wheel_smoke, sdist_smoke):
        assert "_assert_support_bundle_privacy" in script_text
        assert "support-bundle-privacy" in script_text
        assert "input_SECRET_TOKEN=secret123.su2" in script_text
        assert "installed support-bundle leaked secret" in script_text
