# NavierTwin Windows Installer Build

This document is the release-build checklist for producing
`NavierTwinSetup.exe` on Windows.

## Requirements

- Windows 10/11 x64
- Python 3.10-3.12 x64
- Inno Setup 6
- Optional: Microsoft SignTool and a code-signing certificate

Build from Windows. PyInstaller does not cross-compile a Windows `.exe` from
Linux/macOS.

## One-command Build

From the repository root:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\build_windows_installer.ps1 -Clean
```

The default `desktop` profile is the size-optimized customer installer. It
bundles CFD import, the GUI, 3D viewing, ROM/surrogate basics, plotting, and
packaging support, but does not bundle heavyweight optional ML/PDF/API stacks
such as PyTorch, SMT, PyDMD, WeasyPrint, FastAPI, or PhysicsNeMo.

To build a larger all-in-one bundle with optional ML/server packages available
in the builder environment:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\build_windows_installer.ps1 -Clean -Profile full
```

Expected outputs:

- `dist\NavierTwin\NavierTwin.exe`
- `installer\Output\NavierTwinSetup.exe`

If dependencies are already installed:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\build_windows_installer.ps1 -Clean -SkipDependencyInstall
```

To build only the PyInstaller app without the final installer:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\build_windows_installer.ps1 -Clean -NoInstaller
```

To validate the script and packaging metadata from a non-Windows host:

```powershell
pwsh -NoProfile -File scripts\build_windows_installer.ps1 -ValidateOnly -SkipDependencyInstall
```

## Manual Build

```powershell
python -m pip install --upgrade pip
python -m pip install -e ".[desktop]" pyinstaller pyinstaller-hooks-contrib
python scripts\installer_smoke.py
$env:NAVIER_TWIN_BUILD_PROFILE = "desktop"
python -m PyInstaller --noconfirm --clean installer\naviertwin.spec
iscc installer\naviertwin.iss
```

## Signed Build

Set the Inno Setup sign command through `NAVIER_TWIN_SIGNTOOL`, or pass
`-SignTool` to the helper script.

```powershell
powershell -ExecutionPolicy Bypass -File scripts\build_windows_installer.ps1 -Clean `
  -SignTool 'signtool sign /fd SHA256 /td SHA256 /tr http://timestamp.digicert.com /a $f'
```

The `$f` placeholder is required by Inno Setup and is replaced with each setup
or uninstaller file path.

## Release Smoke

After installing `installer\Output\NavierTwinSetup.exe` on a clean Windows
machine:

1. Launch `NavierTwin` from the Start menu.
2. Open a small `.vtu` file.
3. Confirm the Analyze 3D viewer renders and rotates with mouse drag.
4. In Model, confirm the workflow order is visible:
   `1. 학습 준비`, `2A/2B/2C 모델 학습`, `3. 검증/비교`,
   `4. 데이터 보강`, `5. 운영 중 업데이트`.
5. Train a small RBF/Kriging demo model.
6. If testing CFD direct-field training, load multiple cases in Import, select
   params CSV/input columns/output fields in Model, then train.
7. Run a Twin prediction and confirm `twin_pred_*` fields appear in the viewer.

## Notes

- The installer script reads `dist\NavierTwin\*`; always run PyInstaller first.
- `installer\naviertwin.iss` version must match `pyproject.toml`.
- Optional heavy packages such as PhysicsNeMo can be installed into the release
  builder environment and built with `-Profile full` if they should be bundled.
- GitHub Releases can host the generated `.exe` installer as a release asset as
  long as the single asset remains under GitHub's release asset size limit.
