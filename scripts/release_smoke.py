"""Run the customer-facing release smoke suite.

This script keeps the commercial readiness check small and deterministic:
packaging metadata, CLI entry points, offscreen GUI wiring, and post-processing
panel integration. It is intentionally separate from the full regression suite.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import zipfile
from hashlib import sha256
from pathlib import Path

from release_versions import project_version

CUSTOMER_SMOKE_TESTS = [
    "tests/test_packaging_metadata.py",
    "tests/test_dependency_policy.py",
    "tests/test_installer_metadata.py",
    "tests/test_ci_workflow.py",
    "tests/test_docs_structure.py",
    "tests/test_readme_quickstart_smoke.py",
    "tests/test_logger.py",
    "tests/test_verification_emit.py",
    "tests/test_version_consistency.py",
    "tests/test_updater.py",
    "tests/test_doctor_cli.py",
    "tests/test_dataset_preflight.py",
    "tests/test_support_bundle_cli.py",
    "tests/test_support_bundle_privacy.py",
    "tests/test_pipeline_demo_cli.py",
    "tests/test_api_server_smoke.py",
    "tests/test_main_cli.py",
    "tests/test_cli_subcommands.py",
    "tests/test_lcs_pgd_entropy_api.py",
    "tests/test_analyze_panel_advanced_gui.py",
    "tests/test_export_panel_report_gui.py",
    "tests/test_import_panel_formats.py",
    "tests/test_main_window_postproc.py",
    "tests/test_main_window_update_check_gui.py",
    "tests/test_model_compare_dashboard_gui.py",
    "tests/test_model_panel_loss_curve_gui.py",
    "tests/test_ntwin_project_open_gui.py",
    "tests/test_post_process_facade.py",
    "tests/test_postproc_panel.py",
]


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _smoke_env() -> dict[str, str]:
    env = dict(os.environ)
    env.setdefault("QT_QPA_PLATFORM", "offscreen")
    env.setdefault("MPLCONFIGDIR", "/tmp/mpl")
    env.setdefault("NAVIER_TWIN_LOG_DIR", "/tmp/naviertwin/logs")
    src = str(_repo_root() / "src")
    env["PYTHONPATH"] = src if not env.get("PYTHONPATH") else f"{src}{os.pathsep}{env['PYTHONPATH']}"
    return env


def _run(args: list[str], *, env: dict[str, str]) -> int:
    print("+", " ".join(args), flush=True)
    return subprocess.run(args, cwd=_repo_root(), env=env, check=False).returncode


def _run_version(args: list[str], *, env: dict[str, str]) -> int:
    print("+", " ".join(args), flush=True)
    result = subprocess.run(
        args,
        cwd=_repo_root(),
        env=env,
        check=False,
        capture_output=True,
        text=True,
    )
    if result.stdout:
        print(result.stdout, end="" if result.stdout.endswith("\n") else "\n")
    if result.stderr:
        print(result.stderr, end="" if result.stderr.endswith("\n") else "\n", file=sys.stderr)
    if result.returncode != 0:
        return result.returncode
    output = result.stdout.strip()
    expected = f"naviertwin {project_version()}"
    if output != expected:
        print(f"version output mismatch: expected {expected!r}, got {output!r}", file=sys.stderr)
        return 1
    return 0


def _run_json(
    args: list[str],
    *,
    env: dict[str, str],
    allowed_statuses: set[str],
) -> int:
    print("+", " ".join(args), flush=True)
    result = subprocess.run(
        args,
        cwd=_repo_root(),
        env=env,
        check=False,
        capture_output=True,
        text=True,
    )
    if result.stdout:
        print(result.stdout, end="" if result.stdout.endswith("\n") else "\n")
    if result.stderr:
        print(result.stderr, end="" if result.stderr.endswith("\n") else "\n", file=sys.stderr)
    if result.returncode != 0:
        return result.returncode

    try:
        payload = json.loads(result.stdout)
    except json.JSONDecodeError as exc:
        print(f"JSON smoke parse failed: {exc}", file=sys.stderr)
        return 1
    status = payload.get("status")
    if status not in allowed_statuses:
        print(f"JSON smoke status mismatch: {status!r}", file=sys.stderr)
        return 1
    return 0


def _run_file_json(
    args: list[str],
    *,
    env: dict[str, str],
    output: Path,
    allowed_statuses: set[str],
) -> int:
    code = _run_json(args, env=env, allowed_statuses=allowed_statuses)
    if code != 0:
        return code
    try:
        payload = json.loads(output.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        print(f"JSON smoke output file parse failed: {exc}", file=sys.stderr)
        return 1
    status = payload.get("status")
    if status not in allowed_statuses:
        print(f"JSON smoke output file status mismatch: {status!r}", file=sys.stderr)
        return 1
    return 0


def _validate_support_bundle_artifacts(payload: dict[str, object], outdir: Path) -> int:
    expected_files = ["doctor.json", "preflight.json", "metadata.json"]
    if payload.get("files") != expected_files:
        print(f"support-bundle files mismatch: {payload.get('files')!r}", file=sys.stderr)
        return 1
    for name in expected_files:
        if not (outdir / name).exists():
            print(f"support-bundle missing artifact: {name}", file=sys.stderr)
            return 1

    try:
        metadata = json.loads((outdir / "metadata.json").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        print(f"support-bundle metadata parse failed: {exc}", file=sys.stderr)
        return 1
    if metadata != payload:
        print("support-bundle metadata.json does not match stdout payload", file=sys.stderr)
        return 1

    artifacts = payload.get("artifacts")
    if not isinstance(artifacts, dict):
        print("support-bundle payload missing artifacts integrity manifest", file=sys.stderr)
        return 1
    for name in ["doctor.json", "preflight.json"]:
        entry = artifacts.get(name)
        if not isinstance(entry, dict):
            print(f"support-bundle artifact manifest missing {name}", file=sys.stderr)
            return 1
        data = (outdir / name).read_bytes()
        if entry.get("bytes") != len(data):
            print(f"support-bundle byte count mismatch for {name}", file=sys.stderr)
            return 1
        if entry.get("sha256") != sha256(data).hexdigest():
            print(f"support-bundle sha256 mismatch for {name}", file=sys.stderr)
            return 1

    zip_path = payload.get("zip_path")
    expected_zip = outdir / "support-bundle.zip"
    if zip_path != str(expected_zip):
        print(f"support-bundle zip_path mismatch: {zip_path!r}", file=sys.stderr)
        return 1
    if not expected_zip.exists():
        print("support-bundle missing support-bundle.zip", file=sys.stderr)
        return 1
    try:
        with zipfile.ZipFile(expected_zip) as zf:
            manifest = json.loads(zf.read("MANIFEST.json"))
    except Exception as exc:  # pragma: no cover - defensive smoke validation
        print(f"support-bundle zip manifest read failed: {exc}", file=sys.stderr)
        return 1
    manifest_map = {str(entry.get("name")): entry for entry in manifest}
    for name in expected_files:
        if name not in manifest_map:
            print(f"support-bundle zip manifest missing {name}", file=sys.stderr)
            return 1
        data = (outdir / name).read_bytes()
        if manifest_map[name].get("bytes") != len(data):
            print(f"support-bundle zip manifest bytes mismatch for {name}", file=sys.stderr)
            return 1
        if manifest_map[name].get("sha256") != sha256(data).hexdigest():
            print(f"support-bundle zip manifest sha256 mismatch for {name}", file=sys.stderr)
            return 1
    return 0


def _run_support_bundle_smoke(*, env: dict[str, str]) -> int:
    outdir = Path("/tmp/naviertwin-support-bundle-smoke")
    command = [
        sys.executable,
        "-m",
        "naviertwin.main",
        "support-bundle",
        "--outdir",
        str(outdir),
        "--preflight",
        "tests/fixtures/tiny_square.su2",
        "--zip",
    ]
    print("+", " ".join(command), flush=True)
    result = subprocess.run(
        command,
        cwd=_repo_root(),
        env=env,
        check=False,
        capture_output=True,
        text=True,
    )
    if result.stdout:
        print(result.stdout, end="" if result.stdout.endswith("\n") else "\n")
    if result.stderr:
        print(result.stderr, end="" if result.stderr.endswith("\n") else "\n", file=sys.stderr)
    if result.returncode != 0:
        return result.returncode
    try:
        payload = json.loads(result.stdout)
    except json.JSONDecodeError as exc:
        print(f"support-bundle JSON parse failed: {exc}", file=sys.stderr)
        return 1
    if payload.get("status") not in {"ok", "warn"}:
        print(f"support-bundle status mismatch: {payload.get('status')!r}", file=sys.stderr)
        return 1
    return _validate_support_bundle_artifacts(payload, outdir)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--skip-collect",
        action="store_true",
        help="Skip full pytest collection safety check.",
    )
    args = parser.parse_args(argv)

    env = _smoke_env()
    commands = [
        *([["naviertwin", "--help"]] if shutil.which("naviertwin") else []),
        [sys.executable, "scripts/installer_smoke.py"],
        [sys.executable, "-m", "naviertwin.main", "--help"],
    ]
    version_commands = [
        *([["naviertwin", "--version"]] if shutil.which("naviertwin") else []),
        [sys.executable, "-m", "naviertwin.main", "--version"],
        [sys.executable, "main.py", "--version"],
    ]
    json_commands = [
        ([sys.executable, "-m", "naviertwin.main", "doctor", "--json"], {"ok", "warn"}),
        (
            [
                sys.executable,
                "-m",
                "naviertwin.main",
                "preflight",
                "tests/fixtures/tiny_square.su2",
                "--json",
            ],
            {"ok"},
        ),
        (
            [
                sys.executable,
                "-m",
                "naviertwin.main",
                "pipeline-demo",
                "--outdir",
                "/tmp/naviertwin-pipeline-demo-smoke",
            ],
            {"ok"},
        ),
    ]
    file_json_commands = [
        (
            [
                sys.executable,
                "scripts/license_report.py",
                "--json",
                "--output",
                "/tmp/naviertwin-license-report-smoke.json",
            ],
            Path("/tmp/naviertwin-license-report-smoke.json"),
            {"ok"},
        ),
    ]
    commands.extend(
        [
            [sys.executable, "main.py", "--help"],
            [sys.executable, "-m", "pytest", "-q", *CUSTOMER_SMOKE_TESTS],
        ]
    )
    if not args.skip_collect:
        commands.append([sys.executable, "-m", "pytest", "--collect-only", "-q"])

    for command in commands:
        code = _run(command, env=env)
        if code != 0:
            return code
        if command == [sys.executable, "-m", "naviertwin.main", "--help"]:
            for version_command in version_commands:
                code = _run_version(version_command, env=env)
                if code != 0:
                    return code
            for json_command, allowed_statuses in json_commands:
                code = _run_json(
                    json_command,
                    env=env,
                    allowed_statuses=allowed_statuses,
                )
                if code != 0:
                    return code
            code = _run_support_bundle_smoke(env=env)
            if code != 0:
                return code
            for json_command, output, allowed_statuses in file_json_commands:
                code = _run_file_json(
                    json_command,
                    env=env,
                    output=output,
                    allowed_statuses=allowed_statuses,
                )
                if code != 0:
                    return code
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
