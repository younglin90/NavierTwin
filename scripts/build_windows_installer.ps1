param(
    [switch]$Clean,
    [switch]$SkipDependencyInstall,
    [switch]$NoInstaller,
    [switch]$ValidateOnly,
    [ValidateSet("desktop", "full")]
    [string]$Profile = "desktop",
    [ValidateSet("nuitka", "pyinstaller")]
    [string]$Backend = "nuitka",
    [string]$Python = "python",
    [string]$Iscc = "",
    [string]$SignTool = ""
)

$ErrorActionPreference = "Stop"

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$RepoRoot = Resolve-Path (Join-Path $ScriptDir "..")
Set-Location $RepoRoot

function Resolve-Iscc {
    param([string]$Requested)
    if ($Requested) {
        return $Requested
    }
    $fromPath = Get-Command "iscc.exe" -ErrorAction SilentlyContinue
    if ($fromPath) {
        return $fromPath.Source
    }
    $default = "${env:ProgramFiles(x86)}\Inno Setup 6\ISCC.exe"
    if (Test-Path $default) {
        return $default
    }
    return ""
}

Write-Host "== NavierTwin Windows installer build =="
Write-Host "Repo: $RepoRoot"
Write-Host "Profile: $Profile"
Write-Host "Backend: $Backend"

& $Python --version

$IsWindowsPlatform = ($PSVersionTable.Platform -eq "Win32NT") -or ($env:OS -eq "Windows_NT")
if ((-not $IsWindowsPlatform) -and (-not $ValidateOnly)) {
    throw "This release build script must run on Windows. Use -ValidateOnly on non-Windows hosts."
}

if ((-not $SkipDependencyInstall) -and (-not $ValidateOnly)) {
    Write-Host "== Installing build/runtime dependencies =="
    & $Python -m pip install --upgrade pip
    $extra = if ($Profile -eq "full") { ".[core]" } else { ".[desktop]" }
    if ($Backend -eq "nuitka") {
        & $Python -m pip install -e $extra nuitka ordered-set zstandard
    }
    else {
        & $Python -m pip install -e $extra pyinstaller pyinstaller-hooks-contrib
    }
}

Write-Host "== Validating installer metadata =="
& $Python scripts/installer_smoke.py

if ($ValidateOnly) {
    Write-Host "ValidateOnly complete. Windows-only packager/Inno build steps were skipped."
    exit 0
}

if ($Clean) {
    Write-Host "== Cleaning previous build outputs =="
    if (Test-Path "build") {
        Remove-Item -Recurse -Force "build"
    }
    if (Test-Path "dist\NavierTwin") {
        Remove-Item -Recurse -Force "dist\NavierTwin"
    }
    if (Test-Path "installer\Output") {
        Remove-Item -Recurse -Force "installer\Output"
    }
}

$env:NAVIER_TWIN_BUILD_PROFILE = $Profile
if ($Backend -eq "nuitka") {
    Write-Host "== Building Nuitka standalone app =="
    $NuitkaBuild = Join-Path $RepoRoot "build\nuitka"
    $NuitkaDist = Join-Path $NuitkaBuild "gui_entry.dist"
    $DistDir = Join-Path $RepoRoot "dist\NavierTwin"
    if (Test-Path $NuitkaBuild) {
        Remove-Item -Recurse -Force $NuitkaBuild
    }
    if (Test-Path $DistDir) {
        Remove-Item -Recurse -Force $DistDir
    }
    $NuitkaArgs = @(
        "-m", "nuitka",
        "--standalone",
        "--assume-yes-for-downloads",
        "--windows-console-mode=disable",
        "--enable-plugin=pyside6",
        "--include-qt-plugins=platforms,styles,imageformats,iconengines,tls,generic",
        "--include-package=naviertwin",
        "--include-package=pyvista",
        "--include-package=pyvistaqt",
        "--include-package=vtkmodules",
        "--include-package=meshio",
        "--include-package=h5py",
        "--include-package=numpy",
        "--include-package=scipy",
        "--include-package=pandas",
        "--include-package=matplotlib",
        "--include-package=sklearn",
        "--include-package=smt",
        "--include-package=pydmd",
        "--include-package=SALib",
        "--include-package=pip",
        "--output-filename=NavierTwin.exe",
        "--output-dir=$NuitkaBuild",
        "--include-data-dir=src\naviertwin\gui\styles=naviertwin\gui\styles",
        "--noinclude-pytest-mode=nofollow",
        "--noinclude-unittest-mode=nofollow",
        "--noinclude-pydoc-mode=nofollow",
        "--noinclude-IPython-mode=nofollow",
        "--noinclude-numba-mode=nofollow",
        "--noinclude-setuptools-mode=nofollow",
        "--nofollow-import-to=*.tests",
        "--nofollow-import-to=*.tests.*",
        "--nofollow-import-to=*.test",
        "--nofollow-import-to=*.test.*",
        "--nofollow-import-to=tests",
        "--nofollow-import-to=tests.*",
        "--nofollow-import-to=test",
        "--nofollow-import-to=test.*",
        "--nofollow-import-to=sklearn.conftest",
        "--nofollow-import-to=smt.utils.test",
        "--nofollow-import-to=smt.utils.test.*"
    )
    if ($Profile -eq "desktop") {
        $NuitkaArgs += "--nofollow-import-to=torch,torch_geometric,torchdiffeq,physicsnemo,onnx,fastapi,uvicorn,weasyprint,shap,captum,trame,trame_client,trame_vtk,trame_vuetify"
    }
    else {
        $NuitkaArgs += @(
            "--include-package=torch",
            "--include-package=onnx",
            "--include-package=fastapi",
            "--include-package=uvicorn",
            "--include-package=weasyprint",
            "--include-package=shap"
        )
    }
    if (Test-Path "resources") {
        $NuitkaArgs += "--include-data-dir=resources=resources"
    }
    $NuitkaArgs += "src\naviertwin\gui_entry.py"
    & $Python @NuitkaArgs
    if (-not (Test-Path $NuitkaDist)) {
        throw "Nuitka output missing: $NuitkaDist"
    }
    New-Item -ItemType Directory -Force -Path $DistDir | Out-Null
    Copy-Item -Recurse -Force (Join-Path $NuitkaDist "*") $DistDir
}
else {
    Write-Host "== Building PyInstaller onedir app =="
    $SpecPath = Join-Path $RepoRoot "installer\naviertwin.spec"
    & $Python -m PyInstaller --noconfirm --clean $SpecPath
}

$Exe = Join-Path $RepoRoot "dist\NavierTwin\NavierTwin.exe"
if (-not (Test-Path $Exe)) {
    throw "Packager output missing: $Exe"
}
Write-Host "Packager output: $Exe"

Write-Host "== Re-validating installer metadata =="
& $Python scripts/installer_smoke.py

if ($NoInstaller) {
    Write-Host "Skipping Inno Setup installer build because -NoInstaller was supplied."
    exit 0
}

$ResolvedIscc = Resolve-Iscc $Iscc
if (-not $ResolvedIscc) {
    throw "Inno Setup 6 ISCC.exe not found. Install Inno Setup 6 or pass -Iscc <path>."
}

if ($SignTool) {
    $env:NAVIER_TWIN_SIGNTOOL = $SignTool
}

Write-Host "== Building Inno Setup installer =="
& $ResolvedIscc installer\naviertwin.iss

$Setup = Join-Path $RepoRoot "installer\Output\NavierTwinSetup.exe"
if (-not (Test-Path $Setup)) {
    throw "Installer output missing: $Setup"
}

Write-Host "Installer output: $Setup"
Write-Host "Done."
