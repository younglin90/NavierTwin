"""런타임 환경 정보 — Python / OS / GPU / 주요 라이브러리 버전.

Examples:
    >>> from naviertwin.utils.env_info import collect_env
    >>> info = collect_env()
    >>> "python" in info
    True
"""

from __future__ import annotations

import platform
import sys
from importlib import metadata
from typing import Any

_DISTRIBUTION_NAMES = {
    "sklearn": "scikit-learn",
}


def _pkg_version(name: str) -> str | None:
    try:
        return metadata.version(_DISTRIBUTION_NAMES.get(name, name))
    except metadata.PackageNotFoundError:
        return None


def collect_env() -> dict[str, Any]:
    info: dict[str, Any] = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
    }
    packages = (
        "numpy", "scipy", "pandas", "sklearn", "torch", "h5py",
        "pyvista", "PySide6", "matplotlib",
    )
    pkg_idx = 0
    while pkg_idx < len(packages):
        pkg = packages[pkg_idx]
        v = _pkg_version(pkg)
        if v:
            info[pkg] = v
        pkg_idx += 1

    # GPU 감지
    info["cuda_available"] = False
    info["cuda_devices"] = []
    try:
        import torch

        if torch.cuda.is_available():
            info["cuda_available"] = True
            info["cuda_version"] = str(torch.version.cuda)
            devices: list[str] = []
            device_idx = 0
            device_count = torch.cuda.device_count()
            while device_idx < device_count:
                devices.append(torch.cuda.get_device_name(device_idx))
                device_idx += 1
            info["cuda_devices"] = devices
    except ImportError:
        pass
    return info


def format_env(info: dict[str, Any]) -> str:
    lines = []
    items = list(info.items())
    idx = 0
    while idx < len(items):
        k, v = items[idx]
        lines.append(f"{k:20s}: {v}")
        idx += 1
    return "\n".join(lines)


__all__ = ["collect_env", "format_env"]
