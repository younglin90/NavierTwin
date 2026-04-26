"""Reproducibility manifest — capture python/lib versions + git SHA + seed.

Examples:
    >>> from naviertwin.utils.repro_manifest import build_manifest
    >>> m = build_manifest(seed=42)
    >>> 'python' in m
    True
"""

from __future__ import annotations

import importlib.metadata as md
import platform
import subprocess
from typing import Any


def _git_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL,
        ).decode().strip()
    except Exception:  # noqa: BLE001
        return "unknown"


def build_manifest(*, seed: int | None = None,
                     packages: list[str] | None = None) -> dict[str, Any]:
    pkgs = packages or ["numpy", "scipy", "torch"]
    versions: dict[str, str] = {}
    for p in pkgs:
        try:
            versions[p] = md.version(p)
        except Exception:  # noqa: BLE001
            versions[p] = "missing"
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "git_sha": _git_sha(),
        "seed": seed,
        "packages": versions,
    }


__all__ = ["build_manifest"]
