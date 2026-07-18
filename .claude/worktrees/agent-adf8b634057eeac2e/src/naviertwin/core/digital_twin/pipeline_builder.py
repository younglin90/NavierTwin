"""Config dict → NavierTwinPipeline 자동 조립.

Examples:
    >>> from naviertwin.core.digital_twin.pipeline_builder import build_pipeline
    >>> cfg = {"reducer_kind": "pod", "n_modes": 3, "surrogate_kind": "rbf"}
    >>> pipe = build_pipeline(cfg)
    >>> pipe.n_modes
    3
"""

from __future__ import annotations

from typing import Any

_DEFAULTS = {
    "reducer_kind": "pod",
    "n_modes": 5,
    "surrogate_kind": "kriging",
}


def build_pipeline(cfg: dict[str, Any]) -> Any:
    from naviertwin.core.digital_twin.pipeline import NavierTwinPipeline

    merged = {**_DEFAULTS, **(cfg or {})}
    return NavierTwinPipeline(
        reducer_kind=merged["reducer_kind"],
        n_modes=int(merged["n_modes"]),
        surrogate_kind=merged["surrogate_kind"],
    )


def validate_config(cfg: dict[str, Any]) -> list[str]:
    """cfg 필드 유효성 경고 리스트."""
    issues: list[str] = []
    if "n_modes" in cfg and (not isinstance(cfg["n_modes"], int) or cfg["n_modes"] < 1):
        issues.append("n_modes must be positive int")
    if "reducer_kind" in cfg and cfg["reducer_kind"] not in {
        "pod",
        "incremental_pod",
        "mrpod",
        "dmd",
        "ae",
        "vae",
    }:
        issues.append(f"unknown reducer_kind: {cfg['reducer_kind']}")
    if "surrogate_kind" in cfg and cfg["surrogate_kind"] not in {
        "kriging", "rbf", "gp", "fno", "deeponet",
    }:
        issues.append(f"unknown surrogate_kind: {cfg['surrogate_kind']}")
    return issues


__all__ = ["build_pipeline", "validate_config"]
