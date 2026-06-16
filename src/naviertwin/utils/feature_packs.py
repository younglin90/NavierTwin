"""Downloadable feature-pack support for heavyweight optional stacks.

Feature packs are ZIP archives that contain a ``manifest.json`` and a ``site/``
directory produced by ``pip install --target``.  The Windows setup wizard can
also install selected packs online with pip directly into the same layout.  The
desktop installer can stay small while installed packs are activated at process
startup through ``sys.path``.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import os
import shutil
import sys
import tempfile
import zipfile
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.request import Request, urlopen

from naviertwin import __version__


@dataclass(frozen=True)
class FeaturePackSpec:
    """Metadata for a downloadable optional feature pack."""

    id: str
    name: str
    description: str
    modules: tuple[str, ...]
    packages: tuple[str, ...]
    # pip 의 ``--extra-index-url`` 로 추가될 인덱스 목록. PyTorch CPU 휠 등
    # 공식 PyPI 가 아닌 위치에서 받아야 하는 경우에만 사용.
    extra_index_urls: tuple[str, ...] = ()

    def asset_name(self, version: str = __version__) -> str:
        """Return the expected GitHub Release asset name."""
        return f"NavierTwinFeaturePack-{self.id}-{version}.zip"


FEATURE_PACKS: dict[str, FeaturePackSpec] = {
    "ml-cpu": FeaturePackSpec(
        id="ml-cpu",
        name="ML/Operator Learning CPU Pack",
        description="PyTorch, ONNX, SHAP, Captum and neural/operator-learning dependencies.",
        modules=("torch", "onnx", "shap", "captum"),
        packages=("torch", "onnx", "shap", "captum"),
        # CPU 휠 인덱스 — GPU 없는 PC 에서도 가볍게 설치되도록.
        extra_index_urls=("https://download.pytorch.org/whl/cpu",),
    ),
    "physicsnemo": FeaturePackSpec(
        id="physicsnemo",
        name="NVIDIA PhysicsNeMo Pack",
        description="PhysicsNeMo integration for physics-informed CFD workflows.",
        modules=("torch", "physicsnemo"),
        # PyPI 패키지명은 ``nvidia-physicsnemo`` (Python import 이름은 ``physicsnemo``).
        # 2.0.0 부터 Python ≥ 3.11 만 지원하므로, 3.10 환경에서는 1.3.0 으로 fallback.
        packages=("torch", "nvidia-physicsnemo"),
        # PyTorch CPU 휠 인덱스 — GPU 없는 PC 에서도 ``torch`` 가 작은 휠로 설치됨.
        extra_index_urls=("https://download.pytorch.org/whl/cpu",),
    ),
    "serving": FeaturePackSpec(
        id="serving",
        name="Serving/API Pack",
        description="FastAPI and uvicorn runtime for local REST serving.",
        modules=("fastapi", "uvicorn"),
        packages=("fastapi", "uvicorn"),
    ),
    "reporting": FeaturePackSpec(
        id="reporting",
        name="PDF Reporting Pack",
        description="WeasyPrint runtime for PDF report generation.",
        modules=("weasyprint",),
        packages=("weasyprint",),
    ),
    "advanced-io-mesh": FeaturePackSpec(
        id="advanced-io-mesh",
        name="Advanced IO/Mesh Pack",
        description="Large optional CFD readers and mesh-processing backends.",
        modules=("pyarrow", "zarr", "xarray", "netCDF4", "gmsh", "pymeshlab"),
        packages=("pyarrow", "zarr", "xarray", "netCDF4", "gmsh", "pymeshlab"),
    ),
}

MODULE_TO_PACK: dict[str, str] = {
    "torch": "ml-cpu",
    "onnx": "ml-cpu",
    "shap": "ml-cpu",
    "captum": "ml-cpu",
    "torch_geometric": "ml-cpu",
    "torchdiffeq": "ml-cpu",
    "pywt": "ml-cpu",
    "physicsnemo": "physicsnemo",
    "fastapi": "serving",
    "uvicorn": "serving",
    "weasyprint": "reporting",
    "pyarrow": "advanced-io-mesh",
    "zarr": "advanced-io-mesh",
    "xarray": "advanced-io-mesh",
    "netCDF4": "advanced-io-mesh",
    "gmsh": "advanced-io-mesh",
    "pymeshlab": "advanced-io-mesh",
}


def default_feature_pack_root() -> Path:
    """Return the user-writable feature-pack install root."""
    override = os.environ.get("NAVIER_TWIN_FEATURE_PACK_DIR")
    if override:
        return Path(override).expanduser()
    if sys.platform == "win32":
        base = os.environ.get("LOCALAPPDATA")
        if base:
            return Path(base) / "NavierTwin" / "feature-packs"
    if sys.platform == "darwin":
        return Path.home() / "Library" / "Application Support" / "NavierTwin" / "feature-packs"
    return Path.home() / ".local" / "share" / "NavierTwin" / "feature-packs"


def system_feature_pack_root() -> Path | None:
    """Return the installer-managed shared feature-pack root, when available."""
    override = os.environ.get("NAVIER_TWIN_SYSTEM_FEATURE_PACK_DIR")
    if override:
        return Path(override).expanduser()
    if sys.platform == "win32":
        base = os.environ.get("ProgramData")
        if base:
            return Path(base) / "NavierTwin" / "feature-packs"
        return Path("C:/ProgramData/NavierTwin/feature-packs")
    return None


def feature_pack_roots(root: Path | None = None) -> list[Path]:
    """Return feature-pack roots in activation priority order."""
    if root is not None:
        return [root]
    roots = [default_feature_pack_root()]
    system_root = system_feature_pack_root()
    if system_root is not None and system_root not in roots:
        roots.append(system_root)
    return roots


def default_release_asset_url(
    pack_id: str,
    *,
    version: str = __version__,
    repository: str = "younglin90/NavierTwin",
) -> str:
    """Return the expected GitHub Release download URL for a feature pack."""
    spec = get_feature_pack_spec(pack_id)
    tag = f"v{version}"
    return (
        f"https://github.com/{repository}/releases/download/"
        f"{tag}/{spec.asset_name(version)}"
    )


def get_feature_pack_spec(pack_id: str) -> FeaturePackSpec:
    """Return a known feature-pack spec or raise ``KeyError``."""
    return FEATURE_PACKS[pack_id]


def recommended_pack_for_modules(modules: tuple[str, ...] | list[str]) -> str | None:
    """Return the first feature pack that can satisfy one of ``modules``."""
    if "physicsnemo" in modules:
        return "physicsnemo"
    for module in modules:
        pack_id = MODULE_TO_PACK.get(module)
        if pack_id:
            return pack_id
    return None


def installed_pack_dir(pack_id: str, root: Path | None = None) -> Path:
    """Return the installation directory for ``pack_id``."""
    return (root or default_feature_pack_root()) / pack_id


def installed_site_dir(pack_id: str, root: Path | None = None) -> Path:
    """Return the installed ``site`` directory for ``pack_id``."""
    return installed_pack_dir(pack_id, root) / "site"


def activate_installed_feature_packs(root: Path | None = None) -> list[Path]:
    """Prepend installed feature-pack ``site`` directories to Python import paths.

    Windows 인스톨러가 elevated 로 ``ProgramData`` 에 설치하면 일반 user 가
    하위 site 디렉토리를 못 읽는 경우가 있어 ``OSError`` (특히
    ``PermissionError``) 를 graceful 하게 무시한다. 그래야 적어도 다른 root 의
    팩들은 정상 활성화된다.
    """
    activated: list[Path] = []
    for pack_root in feature_pack_roots(root):
        try:
            if not pack_root.exists():
                continue
        except OSError:
            continue
        # glob 자체가 PermissionError 를 raise 할 수 있어서 iterdir 로 풀어서 처리.
        try:
            children = list(pack_root.iterdir())
        except OSError:
            continue
        for child in sorted(children):
            try:
                site_dir = child / "site"
                if not site_dir.is_dir():
                    continue
            except OSError:
                continue
            site_text = str(site_dir)
            if site_text not in sys.path:
                sys.path.insert(0, site_text)
            _activate_dll_paths(site_dir)
            activated.append(site_dir)
    return activated


def _activate_dll_paths(site_dir: Path) -> None:
    """Expose common native-library directories for feature-pack packages."""
    candidates = [site_dir, site_dir / "torch" / "lib", site_dir / "nvidia"]
    existing = [path for path in candidates if path.exists()]
    if sys.platform == "win32" and hasattr(os, "add_dll_directory"):
        handles = []
        for path in existing:
            try:
                handles.append(os.add_dll_directory(str(path)))  # type: ignore[attr-defined]
            except OSError:
                continue
        # Keep directory handles alive for the process lifetime.
        globals().setdefault("_DLL_DIRECTORY_HANDLES", []).extend(handles)
    current_path = os.environ.get("PATH", "")
    for path in reversed(existing):
        path_text = str(path)
        if path_text not in current_path.split(os.pathsep):
            current_path = f"{path_text}{os.pathsep}{current_path}" if current_path else path_text
    os.environ["PATH"] = current_path


def _path_is_dir_safe(path: Path) -> bool:
    """``path.is_dir()`` but returns False on PermissionError instead of raising."""
    try:
        return path.is_dir()
    except OSError:
        return False


def feature_pack_status(pack_id: str, root: Path | None = None) -> dict[str, Any]:
    """Return installation and module availability status for one pack.

    site 디렉토리가 권한 부족으로 읽히지 않는 경우 ``unreadable_paths`` 와
    ``readable=False`` 가 함께 표시되어 UI 가 명확한 안내를 띄울 수 있다.
    """
    spec = get_feature_pack_spec(pack_id)
    pack_dirs = [installed_pack_dir(pack_id, candidate) for candidate in feature_pack_roots(root)]

    installed_dirs: list[Path] = []
    unreadable_dirs: list[str] = []
    for pack_dir in pack_dirs:
        site_dir = pack_dir / "site"
        manifest_path = pack_dir / "manifest.json"
        # site/manifest 의 존재 검사 자체가 PermissionError 가능 → 권한 부족 표시.
        try:
            site_ok = site_dir.is_dir()
            manifest_ok = manifest_path.exists()
        except PermissionError:
            unreadable_dirs.append(str(pack_dir))
            continue
        except OSError:
            continue
        if site_ok and manifest_ok:
            installed_dirs.append(pack_dir)

    pack_dir = installed_dirs[0] if installed_dirs else pack_dirs[0]
    manifest_path = pack_dir / "manifest.json"
    modules = {module: _module_available(module) for module in spec.modules}
    manifest: dict[str, Any] | None = None
    try:
        if manifest_path.exists():
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        manifest = {"error": "invalid manifest"}
    except OSError:
        manifest = None
    return {
        "id": spec.id,
        "name": spec.name,
        "installed": bool(installed_dirs),
        "readable": not unreadable_dirs or bool(installed_dirs),
        "path": str(pack_dir),
        "installed_paths": [str(path) for path in installed_dirs],
        "unreadable_paths": unreadable_dirs,
        "manifest": manifest,
        "modules": modules,
        "missing_modules": [name for name, ok in modules.items() if not ok],
        "download_url": default_release_asset_url(pack_id),
    }


def all_feature_pack_statuses(root: Path | None = None) -> list[dict[str, Any]]:
    """Return status dictionaries for all known feature packs."""
    return [feature_pack_status(pack_id, root) for pack_id in FEATURE_PACKS]


def download_feature_pack(
    pack_id: str,
    *,
    url: str | None = None,
    output_dir: Path | None = None,
    expected_sha256: str | None = None,
) -> Path:
    """Download a feature-pack ZIP and optionally verify SHA256."""
    spec = get_feature_pack_spec(pack_id)
    download_url = url or default_release_asset_url(pack_id)
    target_dir = output_dir or default_feature_pack_root() / "_downloads"
    target_dir.mkdir(parents=True, exist_ok=True)
    target = target_dir / Path(download_url).name
    if not target.name.endswith(".zip"):
        target = target_dir / spec.asset_name()

    request = Request(download_url, headers={"User-Agent": "NavierTwin-feature-pack"})
    with urlopen(request, timeout=180) as response:
        with target.open("wb") as handle:
            shutil.copyfileobj(response, handle)
    if expected_sha256:
        _verify_sha256(target, expected_sha256)
    return target


def install_feature_pack_archive(
    archive_path: Path | str,
    *,
    root: Path | None = None,
    expected_pack_id: str | None = None,
    expected_sha256: str | None = None,
) -> dict[str, Any]:
    """Install a downloaded feature-pack ZIP into the user feature-pack root."""
    archive = Path(archive_path)
    if expected_sha256:
        _verify_sha256(archive, expected_sha256)

    with zipfile.ZipFile(archive) as zf:
        names = set(zf.namelist())
        if "manifest.json" not in names:
            raise ValueError("feature pack archive missing manifest.json")
        manifest = json.loads(zf.read("manifest.json").decode("utf-8"))
        pack_id = str(manifest.get("id", "")).strip()
        if not pack_id:
            raise ValueError("feature pack manifest missing id")
        if expected_pack_id and pack_id != expected_pack_id:
            raise ValueError(f"feature pack id mismatch: {pack_id} != {expected_pack_id}")
        if pack_id not in FEATURE_PACKS:
            raise ValueError(f"unknown feature pack id: {pack_id}")
        if not any(name.startswith("site/") for name in names):
            raise ValueError("feature pack archive missing site/ payload")

        install_dir = installed_pack_dir(pack_id, root)
        pack_root = install_dir.parent
        pack_root.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix=f"{pack_id}-", dir=str(pack_root)) as tmp:
            temp_dir = Path(tmp)
            zf.extractall(temp_dir)
            target = install_dir
            backup = install_dir.with_suffix(".old")
            if backup.exists():
                shutil.rmtree(backup)
            if target.exists():
                target.rename(backup)
            try:
                shutil.move(str(temp_dir), str(target))
            except Exception:
                if target.exists():
                    shutil.rmtree(target)
                if backup.exists():
                    backup.rename(target)
                raise
            if backup.exists():
                shutil.rmtree(backup)
    activate_installed_feature_packs(root)
    return feature_pack_status(pack_id, root)


def install_feature_pack_online(
    pack_id: str,
    *,
    root: Path | None = None,
    packages: tuple[str, ...] | list[str] | None = None,
    log_file: Path | str | None = None,
    extra_index_urls: tuple[str, ...] | list[str] | None = None,
) -> dict[str, Any]:
    """Install a feature pack online with pip into the feature-pack layout."""
    spec = get_feature_pack_spec(pack_id)
    package_list = tuple(packages or spec.packages)
    index_urls = tuple(
        extra_index_urls if extra_index_urls is not None else spec.extra_index_urls
    )
    install_dir = installed_pack_dir(pack_id, root)
    pack_root = install_dir.parent
    pack_root.mkdir(parents=True, exist_ok=True)
    log_path = Path(log_file) if log_file else None

    _write_install_log(log_path, f"Installing feature pack '{pack_id}' with pip")
    _write_install_log(log_path, f"Packages: {', '.join(package_list)}")
    if index_urls:
        _write_install_log(log_path, f"Extra index URLs: {', '.join(index_urls)}")
    with tempfile.TemporaryDirectory(prefix=f"{pack_id}-", dir=str(pack_root)) as tmp:
        temp_dir = Path(tmp)
        site_dir = temp_dir / "site"
        site_dir.mkdir()
        exit_code = _run_pip_install(package_list, site_dir, log_path, index_urls)
        if exit_code != 0:
            raise RuntimeError(f"pip install failed for feature pack '{pack_id}' ({exit_code})")

        manifest = build_archive_manifest(pack_id)
        manifest.update(
            {
                "install_method": "online-pip",
                "packages": list(package_list),
                "installed_at": datetime.now(timezone.utc).isoformat(),
            }
        )
        (temp_dir / "manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True),
            encoding="utf-8",
        )

        backup = install_dir.with_suffix(".old")
        if backup.exists():
            shutil.rmtree(backup)
        if install_dir.exists():
            install_dir.rename(backup)
        try:
            shutil.move(str(temp_dir), str(install_dir))
        except Exception:
            if install_dir.exists():
                shutil.rmtree(install_dir)
            if backup.exists():
                backup.rename(install_dir)
            raise
        if backup.exists():
            shutil.rmtree(backup)

    _grant_users_read_acl(install_dir, log_path)
    activate_installed_feature_packs(root)
    status = feature_pack_status(pack_id, root)
    _write_install_log(log_path, f"Installed feature pack '{pack_id}' to {install_dir}")
    return status


def _grant_users_read_acl(target: Path, log_path: Path | None = None) -> None:
    """Windows 에서 ``target`` 과 모든 자식에 BUILTIN\\Users 그룹의 read+execute
    권한을 부여한다.

    인스톨러가 elevated 로 ProgramData 에 만든 디렉토리가 SYSTEM/Administrators
    전용 ACL 로 잠겨 일반 user GUI 에서 import 못 하는 문제를 self-heal.
    Windows 외 OS, icacls 미존재, target 미존재 시 silent noop.
    """
    if sys.platform != "win32":
        return
    if not target.exists():
        return
    import subprocess  # noqa: PLC0415

    # SID S-1-5-32-545 = BUILTIN\Users (언어팩 무관). (OI)(CI)=상속, RX=ReadAndExecute.
    cmd = [
        "icacls",
        str(target),
        "/grant",
        "*S-1-5-32-545:(OI)(CI)RX",
        "/T",
        "/C",
        "/Q",
    ]
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=60,
        )
        if log_path is not None:
            _write_install_log(
                log_path,
                f"icacls grant Users:RX → exit {result.returncode}",
            )
    except (FileNotFoundError, subprocess.SubprocessError, OSError):
        # icacls 없거나 timeout — 무시 (사용자가 GUI 의 재설치로 LOCALAPPDATA fallback 가능).
        return


def _run_pip_install(
    packages: tuple[str, ...],
    target_site: Path,
    log_file: Path | None = None,
    extra_index_urls: tuple[str, ...] = (),
) -> int:
    """Run pip in-process so PyInstaller builds can install optional packs.

    콘솔이 attach 되어 있으면 pip 의 진행률 막대가 사용자에게 실시간 표시되며,
    동시에 모든 출력이 ``log_file`` 에도 tee 된다. 콘솔이 없으면 로그 파일에만
    기록한다 (기존 동작 유지).
    """
    _patch_pip_distlib_resource_finder()

    from pip._internal.cli.main import main as pip_main

    args = [
        "install",
        "--upgrade",
        "--no-warn-script-location",
        "--progress-bar",
        "on",  # 진행 막대 강제 활성화 (TTY 아닐 때도).
        "--target",
        str(target_site),
    ]
    for url in extra_index_urls:
        args.extend(["--extra-index-url", url])
    args.extend(packages)
    if log_file is None:
        return int(pip_main(args) or 0)

    log_file.parent.mkdir(parents=True, exist_ok=True)
    # 콘솔이 사용 가능하면 stdout 에 그대로 + 로그 파일에도 기록 (tee).
    # 콘솔이 없으면 로그 파일에만.
    if _has_visible_console():
        tee = _TeeWriter(sys.stdout, log_file)
        with contextlib.redirect_stdout(tee), contextlib.redirect_stderr(tee):
            try:
                return int(pip_main(args) or 0)
            finally:
                tee.close_log()
    else:
        with log_file.open("a", encoding="utf-8") as handle:
            with contextlib.redirect_stdout(handle), contextlib.redirect_stderr(handle):
                return int(pip_main(args) or 0)


def _has_visible_console() -> bool:
    """현재 프로세스에 사용자 가시 콘솔이 attach 되어 있는지 보수적 판단."""
    if sys.platform == "win32":
        try:
            import ctypes

            return bool(ctypes.windll.kernel32.GetConsoleWindow())
        except Exception:
            return False
    try:
        return bool(sys.stdout and sys.stdout.isatty())
    except Exception:
        return False


class _TeeWriter:
    """sys.stdout 과 log 파일에 동시에 쓰는 file-like wrapper."""

    def __init__(self, console_stream: Any, log_path: Path) -> None:
        self._console = console_stream
        self._log_handle = log_path.open("a", encoding="utf-8")

    def write(self, text: str) -> int:
        try:
            self._console.write(text)
            self._console.flush()
        except Exception:
            pass
        try:
            self._log_handle.write(text)
            self._log_handle.flush()
        except Exception:
            pass
        return len(text)

    def flush(self) -> None:
        try:
            self._console.flush()
        except Exception:
            pass
        try:
            self._log_handle.flush()
        except Exception:
            pass

    def isatty(self) -> bool:
        # pip 의 progress bar 가 활성화되려면 isatty=True 가 필요.
        try:
            return bool(self._console.isatty())
        except Exception:
            return False

    def close_log(self) -> None:
        try:
            self._log_handle.close()
        except Exception:
            pass


def _patch_pip_distlib_resource_finder() -> None:
    """Let vendored pip/distlib find wrapper resources in PyInstaller builds."""
    if not getattr(sys, "frozen", False):
        return
    try:
        import pip._vendor.distlib as distlib
        from pip._vendor.distlib import resources
    except Exception:
        return
    loader = getattr(distlib, "__loader__", None)
    if loader is not None:
        resources.register_finder(loader, resources.ResourceFinder)


def _write_install_log(log_file: Path | None, message: str) -> None:
    """Append a timestamped installer log line if a log path was provided."""
    if log_file is None:
        return
    log_file.parent.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).isoformat()
    with log_file.open("a", encoding="utf-8") as handle:
        handle.write(f"{timestamp} {message}\n")


def build_archive_manifest(pack_id: str, *, version: str = __version__) -> dict[str, Any]:
    """Return the manifest that build scripts should place in a feature-pack ZIP."""
    spec = get_feature_pack_spec(pack_id)
    payload = asdict(spec)
    payload.update(
        {
            "version": version,
            "layout": "site",
            "application": "NavierTwin",
        }
    )
    return payload


def _verify_sha256(path: Path, expected: str) -> None:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(chunk)
    digest = hasher.hexdigest()
    if digest.lower() != expected.lower():
        raise ValueError(f"SHA256 mismatch for {path}: {digest} != {expected}")


def _module_available(module: str) -> bool:
    import importlib.util

    try:
        return importlib.util.find_spec(module) is not None
    except (ImportError, ModuleNotFoundError, ValueError):
        return False


__all__ = [
    "FEATURE_PACKS",
    "FeaturePackSpec",
    "activate_installed_feature_packs",
    "all_feature_pack_statuses",
    "build_archive_manifest",
    "default_feature_pack_root",
    "default_release_asset_url",
    "download_feature_pack",
    "feature_pack_status",
    "feature_pack_roots",
    "get_feature_pack_spec",
    "install_feature_pack_archive",
    "installed_pack_dir",
    "installed_site_dir",
    "install_feature_pack_online",
    "recommended_pack_for_modules",
    "system_feature_pack_root",
]
