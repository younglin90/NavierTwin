"""Build and inspect a NavierTwin source distribution artifact.

The check avoids dependency installation and network access. It builds an sdist
from the current tree, then verifies that customer-facing source artifacts
include project metadata, runtime assets, and the release smoke scripts.
"""

from __future__ import annotations

import argparse
import configparser
import json
import os
import shutil
import subprocess
import sys
import tarfile
import tempfile
import venv
import zipfile
from email.parser import Parser
from hashlib import sha256
from pathlib import Path

from release_versions import example_latest_release_version, project_version
from setuptools import build_meta

REQUIRED_SDIST_FILES = [
    "LICENSE",
    "README.md",
    "pyproject.toml",
    "MANIFEST.in",
    "examples/release-metadata.example.json",
    "scripts/release_smoke.py",
    "scripts/installer_smoke.py",
    "scripts/wheel_smoke.py",
    "scripts/sdist_smoke.py",
    "scripts/license_report.py",
    "src/naviertwin/gui/styles/dark_theme.qss",
    "src/naviertwin/gui/styles/i18n/ko.json",
    "src/naviertwin/gui/styles/i18n/en.json",
]

REQUIRED_PKG_INFO_SNIPPETS = [
    "Classifier: Development Status :: 4 - Beta",
]

EXPECTED_INSTALLED_COMMANDS = [
    "benchmark",
    "server",
    "pipeline",
    "pipeline-demo",
    "preflight",
    "support-bundle",
    "autorefine",
    "update-check",
    "doctor",
]


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _enter_installed_runtime(venv_dir: Path) -> Path:
    """Run installed CLI checks outside the repo without inherited PYTHONPATH."""
    runtime_cwd = venv_dir / "runtime-cwd"
    runtime_cwd.mkdir(parents=True, exist_ok=True)
    os.environ.pop("PYTHONPATH", None)
    os.environ["PYTHONNOUSERSITE"] = "1"
    os.chdir(runtime_cwd)
    return runtime_cwd


def _assert_import_from_install(python: Path, venv_dir: Path) -> None:
    import_cmd = [
        str(python),
        "-c",
        (
            "import importlib, json, pathlib; "
            "modules = ['naviertwin', 'naviertwin.core', 'naviertwin.utils']; "
            "print(json.dumps({"
            "name: str(pathlib.Path(importlib.import_module(name).__file__).resolve()) "
            "for name in modules"
            "}))"
        ),
    ]
    print("+", " ".join(import_cmd), flush=True)
    result = subprocess.run(import_cmd, check=True, capture_output=True, text=True)
    sources = {
        name: Path(source).resolve()
        for name, source in json.loads(result.stdout).items()
    }
    repo = _repo_root().resolve()
    install_root = venv_dir.resolve()
    for module_name, source in sources.items():
        if source == repo or repo in source.parents:
            raise RuntimeError(
                f"installed import for {module_name} resolved to repo checkout: {source}"
            )
        if install_root not in source.parents:
            raise RuntimeError(
                f"installed import for {module_name} did not resolve inside venv: {source}"
            )


def _assert_installed_version(naviertwin: Path) -> None:
    version_cmd = [str(naviertwin), "--version"]
    print("+", " ".join(version_cmd), flush=True)
    result = subprocess.run(
        version_cmd,
        check=True,
        capture_output=True,
        text=True,
    )
    expected = f"naviertwin {project_version()}"
    actual = result.stdout.strip()
    if actual != expected:
        raise RuntimeError(f"installed version output mismatch: expected {expected!r}, got {actual!r}")


def _assert_support_bundle_outputs(
    payload: dict[str, object],
    support_dir: Path,
    expected_files: list[str],
    allowed_statuses: set[str] | None = None,
) -> None:
    statuses = allowed_statuses or {"ok", "warn"}
    if payload.get("status") not in statuses:
        raise RuntimeError(f"installed support-bundle returned unexpected payload: {payload}")
    if payload.get("files") != expected_files:
        raise RuntimeError(f"installed support-bundle payload missing files: {payload}")
    missing = [name for name in expected_files if not (support_dir / name).exists()]
    if missing:
        raise RuntimeError(f"installed support-bundle did not create expected artifacts: {missing}")
    zip_path = payload.get("zip_path")
    expected_zip = support_dir / "support-bundle.zip"
    if zip_path is not None:
        if zip_path != str(expected_zip):
            raise RuntimeError(f"installed support-bundle zip_path mismatch: {zip_path!r}")
        if not expected_zip.exists():
            raise RuntimeError("installed support-bundle zip is missing")
        with zipfile.ZipFile(expected_zip) as zf:
            manifest = json.loads(zf.read("MANIFEST.json"))
        manifest_map = {str(entry.get('name')): entry for entry in manifest}
        for name in expected_files:
            if name not in manifest_map:
                raise RuntimeError(f"installed support-bundle zip manifest missing {name}")
            data = (support_dir / name).read_bytes()
            if manifest_map[name].get("bytes") != len(data):
                raise RuntimeError(f"installed support-bundle zip manifest bytes mismatch for {name}")
            if manifest_map[name].get("sha256") != sha256(data).hexdigest():
                raise RuntimeError(
                    f"installed support-bundle zip manifest sha256 mismatch for {name}"
                )


def _assert_support_bundle_privacy(support_dir: Path, expected_files: list[str]) -> None:
    forbidden = ["secret123", "SECRET_TOKEN=secret123"]
    for name in expected_files:
        text = (support_dir / name).read_text(encoding="utf-8")
        for value in forbidden:
            if value in text:
                raise RuntimeError(f"installed support-bundle leaked secret in {name}")
    zip_path = support_dir / "support-bundle.zip"
    if zip_path.exists():
        with zipfile.ZipFile(zip_path) as zf:
            for name in expected_files:
                text = zf.read(name).decode("utf-8")
                for value in forbidden:
                    if value in text:
                        raise RuntimeError(f"installed support-bundle zip leaked secret in {name}")


def _build_sdist(outdir: Path) -> Path:
    if outdir.exists():
        shutil.rmtree(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    print("+ setuptools.build_meta.build_sdist", outdir, flush=True)
    previous_cwd = Path.cwd()
    os.chdir(_repo_root())
    try:
        sdist_name = build_meta.build_sdist(str(outdir))
    finally:
        os.chdir(previous_cwd)

    sdists = sorted(outdir.glob("naviertwin-*.tar.gz"))
    expected = outdir / sdist_name
    if expected not in sdists:
        raise RuntimeError(f"build backend returned {expected}, but found {sdists}")
    if len(sdists) != 1:
        raise RuntimeError(f"expected exactly one naviertwin sdist, got {sdists}")
    return sdists[0]


def _strip_root(member_name: str) -> str:
    parts = Path(member_name).parts
    return str(Path(*parts[1:])) if len(parts) > 1 else member_name


def _validate_sdist(sdist: Path) -> None:
    with tarfile.open(sdist, "r:gz") as tf:
        names = {_strip_root(member.name) for member in tf.getmembers() if member.isfile()}
        missing = [name for name in REQUIRED_SDIST_FILES if name not in names]
        if missing:
            raise RuntimeError(f"sdist is missing required files: {missing}")

        pkg_info_members = [
            member for member in tf.getmembers() if _strip_root(member.name) == "PKG-INFO"
        ]
        if len(pkg_info_members) != 1:
            raise RuntimeError(f"expected one PKG-INFO file, got {pkg_info_members}")
        pkg_info_file = tf.extractfile(pkg_info_members[0])
        if pkg_info_file is None:
            raise RuntimeError("unable to read PKG-INFO from sdist")
        pkg_info = pkg_info_file.read().decode("utf-8")
        missing_meta = [
            snippet for snippet in REQUIRED_PKG_INFO_SNIPPETS if snippet not in pkg_info
        ]
        if missing_meta:
            raise RuntimeError(f"sdist PKG-INFO missing snippets: {missing_meta}")
        message = Parser().parsestr(pkg_info)
        if message.get("Name") != "naviertwin":
            raise RuntimeError(f"sdist Name mismatch: {message.get('Name')}")
        if message.get("Requires-Python") != ">=3.10":
            raise RuntimeError(f"sdist Requires-Python mismatch: {message.get('Requires-Python')}")
        if message.get("License-Expression") != "MIT":
            raise RuntimeError(
                f"sdist License-Expression mismatch: {message.get('License-Expression')}"
            )
        license_files = set(message.get_all("License-File", []))
        if "LICENSE" not in license_files:
            raise RuntimeError(f"sdist License-File missing LICENSE: {license_files}")
        extras = set(message.get_all("Provides-Extra", []))
        if extras != {"core", "full", "dev"}:
            raise RuntimeError(f"sdist Provides-Extra mismatch: {extras}")
        requires_dist = message.get_all("Requires-Dist", [])
        required_requires_dist_snippets = [
            "PySide6>=6.6; extra == \"core\"",
            "naviertwin[core]; extra == \"full\"",
            "pytest>=8.1; extra == \"dev\"",
            "tomli>=2.0; python_version < \"3.11\" and extra == \"dev\"",
        ]
        for snippet in required_requires_dist_snippets:
            if snippet not in requires_dist:
                raise RuntimeError(
                    f"sdist Requires-Dist missing '{snippet}': {requires_dist}"
                )

        entry_points_members = [
            member
            for member in tf.getmembers()
            if _strip_root(member.name) == "src/naviertwin.egg-info/entry_points.txt"
        ]
        if len(entry_points_members) != 1:
            raise RuntimeError(f"expected one egg-info entry_points.txt, got {entry_points_members}")
        entry_points_file = tf.extractfile(entry_points_members[0])
        if entry_points_file is None:
            raise RuntimeError("unable to read egg-info entry_points.txt from sdist")
        entry_points = entry_points_file.read().decode("utf-8")
        parser = configparser.ConfigParser()
        parser.read_string(entry_points)
        sections = set(parser.sections())
        if sections != {"console_scripts"}:
            raise RuntimeError(f"sdist entry_points sections mismatch: {sections}")
        console_scripts = {name: value for name, value in parser.items("console_scripts")}
        if console_scripts != {"naviertwin": "naviertwin.main:main"}:
            raise RuntimeError(f"sdist console_scripts mismatch: {console_scripts}")

        requires_members = [
            member
            for member in tf.getmembers()
            if _strip_root(member.name) == "src/naviertwin.egg-info/requires.txt"
        ]
        if len(requires_members) != 1:
            raise RuntimeError(f"expected one egg-info requires.txt, got {requires_members}")
        requires_file = tf.extractfile(requires_members[0])
        if requires_file is None:
            raise RuntimeError("unable to read egg-info requires.txt from sdist")
        requires_text = requires_file.read().decode("utf-8")
        for section in ["[core]", "[full]", "[dev]", "[dev:python_version < \"3.11\"]"]:
            if section not in requires_text:
                raise RuntimeError(f"sdist requires.txt missing section {section}")
        if "tomli>=2.0" not in requires_text:
            raise RuntimeError("sdist requires.txt missing tomli>=2.0 marker fallback")
    _validate_license_report_from_sdist(sdist)


def _extract_required_member(tf: tarfile.TarFile, member_name: str, target: Path) -> None:
    members = [member for member in tf.getmembers() if _strip_root(member.name) == member_name]
    if len(members) != 1:
        raise RuntimeError(f"expected one {member_name} in sdist, got {members}")
    source = tf.extractfile(members[0])
    if source is None:
        raise RuntimeError(f"unable to read {member_name} from sdist")
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(source.read())


def _validate_license_report_from_sdist(sdist: Path) -> None:
    with tempfile.TemporaryDirectory(prefix="naviertwin-sdist-license-") as tmp:
        source_root = Path(tmp) / "source"
        with tarfile.open(sdist, "r:gz") as tf:
            _extract_required_member(tf, "pyproject.toml", source_root / "pyproject.toml")
            _extract_required_member(
                tf,
                "scripts/license_report.py",
                source_root / "scripts" / "license_report.py",
            )

        output = source_root / "license-report.json"
        cmd = [
            sys.executable,
            str(source_root / "scripts" / "license_report.py"),
            "--json",
            "--output",
            str(output),
        ]
        print("+", " ".join(cmd), flush=True)
        result = subprocess.run(
            cmd,
            cwd=source_root,
            check=True,
            capture_output=True,
            text=True,
        )
        stdout_payload = json.loads(result.stdout)
        file_payload = json.loads(output.read_text(encoding="utf-8"))
        if stdout_payload != file_payload:
            raise RuntimeError("sdist license_report stdout and output file differ")
        if file_payload.get("status") != "ok":
            raise RuntimeError(f"sdist license_report returned unexpected payload: {file_payload}")


def _build_wheel_from_sdist(sdist: Path, outdir: Path) -> Path:
    """Build an installable wheel from the sdist before clean runtime checks."""
    if outdir.exists():
        shutil.rmtree(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable,
        "-m",
        "pip",
        "wheel",
        str(sdist),
        "--no-deps",
        "--no-index",
        "--no-build-isolation",
        "-w",
        str(outdir),
    ]
    print("+", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)
    wheels = sorted(outdir.glob("naviertwin-*.whl"))
    if len(wheels) != 1:
        raise RuntimeError(f"expected one wheel built from sdist, got {wheels}")
    return wheels[0]


def _install_and_run_cli(artifact: Path, venv_dir: Path) -> None:
    if venv_dir.exists():
        shutil.rmtree(venv_dir)
    venv.EnvBuilder(with_pip=True).create(venv_dir)

    bin_dir = venv_dir / ("Scripts" if sys.platform == "win32" else "bin")
    python = bin_dir / ("python.exe" if sys.platform == "win32" else "python")
    naviertwin = bin_dir / ("naviertwin.exe" if sys.platform == "win32" else "naviertwin")

    install_cmd = [
        str(python),
        "-m",
        "pip",
        "install",
        "--force-reinstall",
        "--no-deps",
        "--no-index",
        str(artifact),
    ]
    print("+", " ".join(install_cmd), flush=True)
    install_env = {**os.environ, "PYTHONNOUSERSITE": "1"}
    install_env.pop("PYTHONPATH", None)
    subprocess.run(install_cmd, check=True, env=install_env)

    _enter_installed_runtime(venv_dir)
    _assert_import_from_install(python, venv_dir)

    help_cmd = [str(naviertwin), "--help"]
    print("+", " ".join(help_cmd), flush=True)
    result = subprocess.run(help_cmd, check=True, capture_output=True, text=True)
    if "NavierTwin" not in result.stdout:
        raise RuntimeError("installed console script help output does not mention NavierTwin")
    for command in EXPECTED_INSTALLED_COMMANDS:
        if command not in result.stdout:
            raise RuntimeError(f"installed console script help output does not expose {command}")
    _assert_installed_version(naviertwin)

    doctor_cmd = [str(naviertwin), "doctor", "--json"]
    print("+", " ".join(doctor_cmd), flush=True)
    doctor_result = subprocess.run(
        doctor_cmd,
        check=True,
        capture_output=True,
        text=True,
    )
    doctor_payload = json.loads(doctor_result.stdout)
    if doctor_payload.get("status") not in {"ok", "warn"}:
        raise RuntimeError(f"installed doctor returned unexpected payload: {doctor_payload}")
    if "checks" not in doctor_payload:
        raise RuntimeError(f"installed doctor payload missing checks: {doctor_payload}")

    doctor_output = venv_dir / "doctor-output.json"
    doctor_output_cmd = [str(naviertwin), "doctor", "--json", "--output", str(doctor_output)]
    print("+", " ".join(doctor_output_cmd), flush=True)
    doctor_output_result = subprocess.run(
        doctor_output_cmd,
        check=True,
        capture_output=True,
        text=True,
    )
    doctor_stdout_payload = json.loads(doctor_output_result.stdout)
    doctor_file_payload = json.loads(doctor_output.read_text(encoding="utf-8"))
    if doctor_stdout_payload != doctor_file_payload:
        raise RuntimeError("installed doctor --output file does not match stdout payload")
    if doctor_file_payload.get("status") not in {"ok", "warn"}:
        raise RuntimeError(f"installed doctor --output returned unexpected payload: {doctor_file_payload}")

    support_dir = venv_dir / "support-bundle"
    support_cmd = [str(naviertwin), "support-bundle", "--outdir", str(support_dir)]
    print("+", " ".join(support_cmd), flush=True)
    support_result = subprocess.run(
        support_cmd,
        check=True,
        capture_output=True,
        text=True,
    )
    support_payload = json.loads(support_result.stdout)
    _assert_support_bundle_outputs(
        payload=support_payload,
        support_dir=support_dir,
        expected_files=["doctor.json", "metadata.json"],
    )

    preflight_fixture = _repo_root() / "tests" / "fixtures" / "tiny_square.su2"
    if not preflight_fixture.exists():
        raise RuntimeError(f"support-bundle preflight fixture is missing: {preflight_fixture}")
    support_preflight_dir = venv_dir / "support-bundle-preflight"
    support_preflight_cmd = [
        str(naviertwin),
        "support-bundle",
        "--outdir",
        str(support_preflight_dir),
        "--preflight",
        str(preflight_fixture),
        "--zip",
    ]
    print("+", " ".join(support_preflight_cmd), flush=True)
    support_preflight_result = subprocess.run(
        support_preflight_cmd,
        check=False,
        capture_output=True,
        text=True,
    )
    support_preflight_payload = json.loads(support_preflight_result.stdout)
    _assert_support_bundle_outputs(
        payload=support_preflight_payload,
        support_dir=support_preflight_dir,
        expected_files=["doctor.json", "preflight.json", "metadata.json"],
        allowed_statuses={"ok", "warn", "error"},
    )

    support_privacy_dir = venv_dir / "support-bundle-privacy"
    secret_preflight = venv_dir / "input_SECRET_TOKEN=secret123.su2"
    secret_acceptance_json = venv_dir / "acceptance_SECRET_TOKEN=secret123.json"
    secret_acceptance_json.write_text(
        json.dumps(
            {
                "status": "ok",
                "package": str(venv_dir / "delivery_SECRET_TOKEN=secret123.zip"),
                "authorization": "Bearer tok.abc.123",
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    secret_acceptance_summary = venv_dir / "acceptance_SECRET_TOKEN=secret123.md"
    secret_acceptance_summary.write_text(
        "# Acceptance\n\nTOKEN=secret123\nAuthorization: Bearer tok.abc.123\n",
        encoding="utf-8",
    )
    support_privacy_cmd = [
        str(naviertwin),
        "support-bundle",
        "--outdir",
        str(support_privacy_dir),
        "--preflight",
        str(secret_preflight),
        "--acceptance-json",
        str(secret_acceptance_json),
        "--acceptance-summary",
        str(secret_acceptance_summary),
        "--zip",
    ]
    print("+", " ".join(support_privacy_cmd), flush=True)
    support_privacy_result = subprocess.run(
        support_privacy_cmd,
        check=False,
        capture_output=True,
        text=True,
    )
    support_privacy_payload = json.loads(support_privacy_result.stdout)
    support_privacy_files = [
        "doctor.json",
        "preflight.json",
        "acceptance.json",
        "acceptance.md",
        "metadata.json",
    ]
    _assert_support_bundle_outputs(
        payload=support_privacy_payload,
        support_dir=support_privacy_dir,
        expected_files=support_privacy_files,
        allowed_statuses={"ok", "warn", "error"},
    )
    _assert_support_bundle_privacy(support_privacy_dir, support_privacy_files)

    preflight_cmd = [
        str(naviertwin),
        "preflight",
        str(venv_dir / "missing-input.su2"),
        "--json",
    ]
    print("+", " ".join(preflight_cmd), flush=True)
    preflight_result = subprocess.run(
        preflight_cmd,
        check=False,
        capture_output=True,
        text=True,
    )
    preflight_payload = json.loads(preflight_result.stdout)
    if preflight_result.returncode != 1:
        raise RuntimeError(
            f"installed preflight missing-path check returned {preflight_result.returncode}"
        )
    if preflight_payload.get("status") != "error":
        raise RuntimeError(f"installed preflight returned unexpected payload: {preflight_payload}")
    if "path_exists" not in preflight_payload.get("errors", []):
        raise RuntimeError(f"installed preflight payload missing path error: {preflight_payload}")

    preflight_output = venv_dir / "preflight-output.json"
    preflight_output_cmd = [
        str(naviertwin),
        "preflight",
        str(venv_dir / "missing-input.su2"),
        "--json",
        "--output",
        str(preflight_output),
    ]
    print("+", " ".join(preflight_output_cmd), flush=True)
    preflight_output_result = subprocess.run(
        preflight_output_cmd,
        check=False,
        capture_output=True,
        text=True,
    )
    preflight_stdout_payload = json.loads(preflight_output_result.stdout)
    preflight_file_payload = json.loads(preflight_output.read_text(encoding="utf-8"))
    if preflight_output_result.returncode != 1:
        raise RuntimeError(
            f"installed preflight --output missing-path check returned "
            f"{preflight_output_result.returncode}"
        )
    if preflight_stdout_payload != preflight_file_payload:
        raise RuntimeError("installed preflight --output file does not match stdout payload")
    if preflight_file_payload.get("status") != "error":
        raise RuntimeError(
            f"installed preflight --output returned unexpected payload: {preflight_file_payload}"
        )

    demo_dir = venv_dir / "pipeline-demo"
    demo_cmd = [str(naviertwin), "pipeline-demo", "--outdir", str(demo_dir)]
    print("+", " ".join(demo_cmd), flush=True)
    demo_result = subprocess.run(
        demo_cmd,
        check=False,
        capture_output=True,
        text=True,
    )
    if demo_result.returncode == 0:
        demo_payload = json.loads(demo_result.stdout)
        if demo_payload.get("status") != "ok":
            raise RuntimeError(f"installed pipeline-demo returned unexpected payload: {demo_payload}")
        if not (demo_dir / "metrics.json").exists() or not (demo_dir / "report.html").exists():
            raise RuntimeError("installed pipeline-demo did not create expected artifacts")
    elif demo_result.returncode == 2:
        combined = demo_result.stdout + demo_result.stderr
        if "pipeline-demo error:" not in demo_result.stderr:
            raise RuntimeError(f"installed pipeline-demo missing clean error: {combined}")
        if "Traceback" in combined:
            raise RuntimeError(f"installed pipeline-demo emitted traceback: {combined}")
    else:
        raise RuntimeError(
            f"installed pipeline-demo returned {demo_result.returncode}: "
            f"{demo_result.stdout}{demo_result.stderr}"
        )

    metadata = _repo_root() / "examples" / "release-metadata.example.json"
    current_version = project_version()
    expected_latest_version = example_latest_release_version()

    update_cmd = [
        str(naviertwin),
        "update-check",
        "--metadata",
        str(metadata),
        "--current-version",
        current_version,
    ]
    print("+", " ".join(update_cmd), flush=True)
    update_result = subprocess.run(
        update_cmd,
        check=True,
        capture_output=True,
        text=True,
    )
    payload = json.loads(update_result.stdout)
    if payload.get("update_available") is not True:
        raise RuntimeError("installed update-check did not report an available update")
    if payload.get("latest_version") != expected_latest_version:
        raise RuntimeError(f"installed update-check returned unexpected payload: {payload}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--outdir",
        default="/tmp/naviertwin-sdist-smoke",
        help="Directory for temporary sdist output.",
    )
    parser.add_argument(
        "--install-smoke",
        action="store_true",
        help="Install the built sdist into a temporary venv and run naviertwin --help.",
    )
    args = parser.parse_args(argv)

    outdir = Path(args.outdir)
    sdist = _build_sdist(outdir)
    _validate_sdist(sdist)
    if args.install_smoke:
        wheel = _build_wheel_from_sdist(sdist, outdir / "wheel-from-sdist")
        _install_and_run_cli(wheel, outdir / "install-venv")
    print(f"sdist smoke passed: {sdist}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
