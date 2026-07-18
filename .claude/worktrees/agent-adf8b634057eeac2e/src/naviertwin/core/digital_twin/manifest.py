"""Pipeline 매니페스트 — 실행 설정/환경/메트릭을 single JSON 으로 스냅샷.

재현 가능한 실험 로그. 다른 팀원에게 공유 용도.

Examples:
    >>> from naviertwin.core.digital_twin.manifest import build_manifest
    >>> m = build_manifest(reducer="pod", n_modes=5, surrogate="rbf", metrics={"rmse": 0.01})
    >>> "timestamp" in m
    True
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from naviertwin.utils.atomic_io import atomic_write_text
from naviertwin.utils.env_info import collect_env
from naviertwin.utils.json_safe import safe_dumps


def build_manifest(
    *,
    reducer: str,
    n_modes: int,
    surrogate: str,
    metrics: dict[str, Any] | None = None,
    extra: dict[str, Any] | None = None,
    include_env: bool = True,
) -> dict[str, Any]:
    """실행 매니페스트 dict."""
    m: dict[str, Any] = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "config": {
            "reducer": reducer,
            "n_modes": int(n_modes),
            "surrogate": surrogate,
        },
        "metrics": dict(metrics or {}),
    }
    if include_env:
        m["environment"] = collect_env()
    if extra:
        m["extra"] = dict(extra)
    return m


def save_manifest(manifest: dict[str, Any], path: str | Path) -> Path:
    p = Path(path)
    atomic_write_text(p, safe_dumps(manifest))
    return p


__all__ = ["build_manifest", "save_manifest"]
