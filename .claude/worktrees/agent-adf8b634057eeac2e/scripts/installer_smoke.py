"""Fast deterministic smoke checks for Windows installer contracts."""

from __future__ import annotations

import json
from pathlib import Path

try:
    import tomllib
except ImportError:  # pragma: no cover - Python 3.10 compatibility
    import tomli as tomllib


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _assert_contains(text: str, needle: str, *, message: str) -> int:
    if needle not in text:
        raise AssertionError(message)
    return 1


def main() -> int:
    root = _repo_root()
    pyproject = tomllib.loads((root / "pyproject.toml").read_text(encoding="utf-8"))
    version = str(pyproject["project"]["version"])

    spec = (root / "installer" / "naviertwin.spec").read_text(encoding="utf-8")
    iss = (root / "installer" / "naviertwin.iss").read_text(encoding="utf-8")

    checks = 0

    checks += _assert_contains(
        iss,
        f'#define NavierTwinVersion "{version}"',
        message="Inno NavierTwinVersion define must match project.version",
    )
    checks += _assert_contains(
        iss,
        "AppVersion={#NavierTwinVersion}",
        message="Inno AppVersion must match project.version",
    )

    checks += _assert_contains(
        spec,
        "ROOT = _resolve_project_root()",
        message="PyInstaller spec must robustly resolve the project root",
    )
    checks += _assert_contains(
        spec,
        'name="NavierTwin"',
        message='PyInstaller EXE/COLLECT name must be "NavierTwin"',
    )
    checks += _assert_contains(
        spec,
        'distpath=str(ROOT / "dist")',
        message='PyInstaller distpath must be str(ROOT / "dist")',
    )
    checks += _assert_contains(
        spec,
        "naviertwin/gui/styles/i18n",
        message="PyInstaller spec must include GUI i18n runtime assets",
    )
    checks += _assert_contains(
        spec,
        "NAVIER_TWIN_BUILD_PROFILE",
        message="PyInstaller spec must support size-optimized build profiles",
    )
    checks += _assert_contains(
        spec,
        'gui_entry.py"',
        message="PyInstaller spec must use the GUI-only entry point",
    )
    checks += _assert_contains(
        spec,
        '"torch"',
        message="PyInstaller spec must explicitly control heavyweight optional packages",
    )
    checks += _assert_contains(
        spec,
        '"smt"',
        message="PyInstaller desktop profile must include essential SMT surrogate backend",
    )
    checks += _assert_contains(
        spec,
        '"pydmd"',
        message="PyInstaller desktop profile must include essential PyDMD ROM backend",
    )
    checks += _assert_contains(
        spec,
        "PySide6.QtWebEngineCore",
        message="PyInstaller spec must exclude unused heavyweight Qt stacks by default",
    )
    checks += _assert_contains(
        spec,
        "_drop_desktop_bundle_item",
        message="PyInstaller spec must prune optional desktop bundle artifacts",
    )
    checks += _assert_contains(
        spec,
        "trame_vtk",
        message="PyInstaller spec must exclude PyVista browser/Jupyter viewer packages",
    )
    checks += _assert_contains(
        spec,
        "PySide6/Qt6Qml.dll",
        message="PyInstaller spec must prune unused Qt QML runtime DLLs",
    )
    checks += _assert_contains(
        spec,
        "matplotlib.backends.backend_qtagg",
        message="PyInstaller spec must include matplotlib Qt backend when installed",
    )
    checks += _assert_contains(
        spec,
        "resources_dir.exists()",
        message="PyInstaller spec must handle optional resources directory",
    )

    checks += _assert_contains(
        iss,
        'Source: "..\\dist\\NavierTwin\\*"',
        message="Inno must package dist/NavierTwin output directory",
    )
    checks += _assert_contains(
        iss,
        "NavierTwin.exe",
        message="Inno must reference NavierTwin.exe",
    )
    checks += _assert_contains(
        iss,
        "OutputBaseFilename=NavierTwinSetup",
        message="Inno OutputBaseFilename must be NavierTwinSetup",
    )
    checks += _assert_contains(
        iss,
        "DefaultDirName={autopf}\\NavierTwin",
        message="Inno DefaultDirName must target Program Files/NavierTwin",
    )
    checks += _assert_contains(
        iss,
        "LicenseFile=..\\LICENSE",
        message="Inno must reference repository LICENSE",
    )
    checks += _assert_contains(
        iss,
        "UninstallDisplayIcon={app}\\NavierTwin.exe",
        message="Inno uninstall icon must point to NavierTwin.exe",
    )
    checks += _assert_contains(
        iss,
        "SetupIconFile=",
        message="Inno SetupIconFile key must exist (may be intentionally blank)",
    )
    checks += _assert_contains(
        iss,
        "AppPublisher=",
        message="Inno AppPublisher key must exist",
    )
    checks += _assert_contains(
        iss,
        "AppPublisherURL=",
        message="Inno AppPublisherURL key must exist",
    )
    checks += _assert_contains(
        iss,
        "NAVIER_TWIN_SIGNTOOL",
        message="Inno must expose env-driven Authenticode SignTool configuration",
    )
    checks += _assert_contains(
        iss,
        "SignTool={#NavierTwinSignTool}",
        message="Inno SignTool must be driven by NAVIER_TWIN_SIGNTOOL",
    )
    checks += _assert_contains(
        iss,
        "SignedUninstaller=yes",
        message="Inno signed builds must sign the uninstaller",
    )
    checks += _assert_contains(
        iss,
        "$f",
        message="Inno SignTool documentation must include the setup-file placeholder",
    )
    checks += _assert_contains(
        iss,
        "featurepacks\\mlcpu",
        message="Inno must ask users whether to install optional ML features",
    )
    checks += _assert_contains(
        iss,
        "--install-feature-pack",
        message="Inno must run NavierTwin's online optional feature installer",
    )
    checks += _assert_contains(
        iss,
        "{commonappdata}\\NavierTwin\\feature-packs",
        message="Inno must install selected optional features into shared app data",
    )
    checks += _assert_contains(
        (root / "src" / "naviertwin" / "gui_entry.py").read_text(encoding="utf-8"),
        "--install-feature-pack",
        message="GUI entry point must support setup-invoked optional feature installation",
    )
    checks += _assert_contains(
        (root / "src" / "naviertwin" / "utils" / "feature_packs.py").read_text(
            encoding="utf-8"
        ),
        "install_feature_pack_online",
        message="Feature-pack utility must support online setup installation",
    )
    checks += _assert_contains(
        spec,
        "pip._internal.cli.main",
        message="Packager must bundle pip for setup-time optional feature installation",
    )
    checks += _assert_contains(
        (root / "scripts" / "build_windows_installer.ps1").read_text(encoding="utf-8"),
        "PyInstaller --noconfirm --clean $SpecPath",
        message="Windows build helper must run PyInstaller with the release spec path",
    )
    checks += _assert_contains(
        (root / "scripts" / "build_windows_installer.ps1").read_text(encoding="utf-8"),
        "[ValidateSet(\"nuitka\", \"pyinstaller\")]",
        message="Windows build helper must expose Nuitka/PyInstaller backends",
    )
    checks += _assert_contains(
        (root / "scripts" / "build_windows_installer.ps1").read_text(encoding="utf-8"),
        "-m\", \"nuitka\"",
        message="Windows build helper must support Nuitka standalone builds",
    )
    checks += _assert_contains(
        (root / "scripts" / "build_windows_installer.ps1").read_text(encoding="utf-8"),
        "ValidateOnly",
        message="Windows build helper must support non-Windows validation mode",
    )
    checks += _assert_contains(
        (root / "scripts" / "build_windows_installer.ps1").read_text(encoding="utf-8"),
        "[ValidateSet(\"desktop\", \"full\")]",
        message="Windows build helper must expose desktop/full build profiles",
    )
    checks += _assert_contains(
        (root / "docs" / "WINDOWS_INSTALLER.md").read_text(encoding="utf-8"),
        "scripts\\build_windows_installer.ps1",
        message="Windows installer documentation must reference the build helper",
    )
    checks += _assert_contains(
        (root / "scripts" / "build_feature_pack.py").read_text(encoding="utf-8"),
        "NavierTwinFeaturePack",
        message="Feature-pack build helper must generate release asset archives",
    )

    publisher = next(
        (line.split("=", 1)[1].strip() for line in iss.splitlines() if line.startswith("AppPublisher=")),
        "",
    )
    publisher_url = next(
        (
            line.split("=", 1)[1].strip()
            for line in iss.splitlines()
            if line.startswith("AppPublisherURL=")
        ),
        "",
    )
    if not publisher:
        raise AssertionError("Inno AppPublisher must be non-empty")
    checks += 1
    if not publisher_url:
        raise AssertionError("Inno AppPublisherURL must be non-empty")
    checks += 1

    dist_dir = root / "dist" / "NavierTwin"
    if dist_dir.exists():
        if not (dist_dir / "NavierTwin.exe").exists():
            raise AssertionError("dist/NavierTwin exists but NavierTwin.exe is missing")
        checks += 1

    print(json.dumps({"status": "ok", "checks": checks}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
