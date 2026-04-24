"""런타임 환경 정보 — Python / OS / GPU / 주요 라이브러리 버전.

Examples:
    >>> from naviertwin.utils.env_info import collect_env
    >>> info = collect_env()
    >>> "python" in info
    True
"""

from __future__ import annotations

import importlib
import platform
import sys
from typing import Any


def _pkg_version(name: str) -> str | None:
    try:
        m = importlib.import_module(name)
        return str(getattr(m, "__version__", "unknown"))
    except ImportError:
        return None


def collect_env() -> dict[str, Any]:
    info: dict[str, Any] = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
    }
    for pkg in ("numpy", "scipy", "pandas", "sklearn", "torch", "h5py",
                "pyvista", "PySide6", "matplotlib"):
        v = _pkg_version(pkg)
        if v:
            info[pkg] = v

    # GPU 감지
    info["cuda_available"] = False
    info["cuda_devices"] = []
    try:
        import torch

        if torch.cuda.is_available():
            info["cuda_available"] = True
            info["cuda_version"] = str(torch.version.cuda)
            info["cuda_devices"] = [
                torch.cuda.get_device_name(i)
                for i in range(torch.cuda.device_count())
            ]
    except ImportError:
        pass
    return info


def format_env(info: dict[str, Any]) -> str:
    lines = []
    for k, v in info.items():
        lines.append(f"{k:20s}: {v}")
    return "\n".join(lines)


__all__ = ["collect_env", "format_env"]
