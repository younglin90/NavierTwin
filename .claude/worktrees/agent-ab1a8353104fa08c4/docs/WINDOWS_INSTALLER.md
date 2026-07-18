# NavierTwin Windows Installer Build

This document is the release-build checklist for producing the Windows desktop
installer and installer-managed optional feature downloads.

## Requirements

- Windows 10/11 x64
- Python 3.10-3.12 x64
- Inno Setup 6
- Optional: Microsoft SignTool and a code-signing certificate

Build from Windows. Python desktop packagers do not cross-compile a reliable
Windows `.exe` from Linux/macOS.

## Packaging Strategy

- `NavierTwinSetup.exe`: desktop installer. It must include the core
  GUI, CFD import, 3D viewer, POD/DMD, SMT-backed surrogate basics, plotting,
  HDF5, and customer diagnostics.
- Optional feature downloads: during Setup, the user can select heavyweight
  stacks to download and install immediately, for example PyTorch/operator
  learning, PhysicsNeMo, API serving, PDF reporting, and large mesh/IO
  backends. Setup calls `NavierTwin.exe --install-feature-pack ...` and installs
  the selected PyPI packages into the feature-pack directory. The customer does
  not need to manually choose ZIP files.
- `-Profile full`: compatibility build for a larger all-in-one bundle when a
  customer explicitly wants every optional dependency inside the installer.

## One-command Build

From the repository root:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\build_windows_installer.ps1 -Clean
```

The default build uses `-Profile desktop -Backend nuitka`. Nuitka is the
preferred customer build because it avoids much of PyInstaller's bootloader
import overhead and usually starts faster for large scientific desktop apps.
It requires a working C/C++ compiler toolchain on the Windows build machine.

If Nuitka is not available or a third-party binary package is problematic, use
the PyInstaller fallback:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\build_windows_installer.ps1 -Clean -Backend pyinstaller
```

The `desktop` profile excludes heavyweight optional ML/PDF/API stacks such as
PyTorch, PhysicsNeMo, WeasyPrint, FastAPI, ONNX, SHAP, Captum, and GNN-specific
packages. It keeps the required desktop stack and ROM/surrogate backends such
as SMT, PyDMD, and SALib.

To build a larger all-in-one bundle with optional ML/server packages available
in the builder environment:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\build_windows_installer.ps1 -Clean -Profile full
```

Expected outputs:

- `dist\NavierTwin\NavierTwin.exe`
- `installer\Output\NavierTwinSetup.exe`

When the generated setup file runs, the "Additional Tasks" page includes
optional feature checkboxes. Checked feature stacks are downloaded from PyPI
and installed into:

```text
%ProgramData%\NavierTwin\feature-packs
```

NavierTwin also keeps supporting per-user packs under `%LOCALAPPDATA%`; at
runtime it activates both locations automatically.

If dependencies are already installed:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\build_windows_installer.ps1 -Clean -SkipDependencyInstall
```

To build only the packaged app without the final installer:

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
python -m pip install -e ".[desktop]" nuitka ordered-set zstandard
python scripts\installer_smoke.py
powershell -ExecutionPolicy Bypass -File scripts\build_windows_installer.ps1 -Clean -SkipDependencyInstall
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

## Installer-Managed Optional Feature Payloads

The normal customer install flow does not require separate ZIP files. The
Setup wizard runs the installed `NavierTwin.exe` in online feature-install mode
for each selected optional task.

For air-gapped support, build optional payload ZIPs on a networked build
machine and deliver them separately to the customer:

```powershell
python scripts\build_feature_pack.py --pack ml-cpu --output-dir dist\feature-packs
python scripts\build_feature_pack.py --pack physicsnemo --output-dir dist\feature-packs
python scripts\build_feature_pack.py --pack serving --output-dir dist\feature-packs
python scripts\build_feature_pack.py --pack reporting --output-dir dist\feature-packs
```

The air-gapped ZIP convention is:

- `NavierTwinFeaturePack-ml-cpu-<version>.zip`
- `NavierTwinFeaturePack-physicsnemo-<version>.zip`
- `NavierTwinFeaturePack-serving-<version>.zip`
- `NavierTwinFeaturePack-reporting-<version>.zip`

The same payloads can still be installed manually through the CLI:

```powershell
NavierTwin.exe feature-pack download --pack ml-cpu --install
NavierTwin.exe feature-pack install --archive NavierTwinFeaturePack-ml-cpu-4.2.58.zip
```

Setup-installed packs live under the shared application data directory.
Manually installed packs live under the user's application data directory. Both
locations are activated automatically when the GUI/CLI starts.

## Notes

- The installer script reads `dist\NavierTwin\*`; always run the selected
  packager first.
- `installer\naviertwin.iss` version must match `pyproject.toml`.
- Optional heavy packages such as PhysicsNeMo should normally ship as Feature
  Pack ZIPs. Use `-Profile full` only for customers who require one large
  all-in-one installer.
- GitHub Releases can host the generated `.exe` installer as a release asset as
  long as the single asset remains under GitHub's release asset size limit.
