param(
    [switch]$Clean,
    [switch]$SkipDependencyInstall,
    [switch]$NoInstaller,
    [switch]$ValidateOnly,
    [ValidateSet("desktop", "full")]
    [string]$Profile = "desktop",
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

& $Python --version

$IsWindowsPlatform = ($PSVersionTable.Platform -eq "Win32NT") -or ($env:OS -eq "Windows_NT")
if ((-not $IsWindowsPlatform) -and (-not $ValidateOnly)) {
    throw "This release build script must run on Windows. Use -ValidateOnly on non-Windows hosts."
}

if ((-not $SkipDependencyInstall) -and (-not $ValidateOnly)) {
    Write-Host "== Installing build/runtime dependencies =="
    & $Python -m pip install --upgrade pip
    $extra = if ($Profile -eq "full") { ".[core]" } else { ".[desktop]" }
    & $Python -m pip install -e $extra pyinstaller pyinstaller-hooks-contrib
}

Write-Host "== Validating installer metadata =="
& $Python scripts/installer_smoke.py

if ($ValidateOnly) {
    Write-Host "ValidateOnly complete. Windows-only PyInstaller/Inno build steps were skipped."
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

Write-Host "== Building PyInstaller onedir app =="
$SpecPath = Join-Path $RepoRoot "installer\naviertwin.spec"
$env:NAVIER_TWIN_BUILD_PROFILE = $Profile
& $Python -m PyInstaller --noconfirm --clean $SpecPath

$Exe = Join-Path $RepoRoot "dist\NavierTwin\NavierTwin.exe"
if (-not (Test-Path $Exe)) {
    throw "PyInstaller output missing: $Exe"
}
Write-Host "PyInstaller output: $Exe"

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
