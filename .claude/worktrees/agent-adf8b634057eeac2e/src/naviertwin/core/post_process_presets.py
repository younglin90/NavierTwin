"""Op별 scalar 파라미터 프리셋 — 사전 정의 + 사용자 저장/로드.

자주 쓰는 파라미터 조합을 프리셋으로 저장. GUI에서 빠른 전환 가능.

Examples:
    >>> from naviertwin.core.post_process_presets import (
    ...     factory_presets, PresetStore
    ... )
    >>> presets = factory_presets("psd_welch")
    >>> "high_resolution" in presets
    True
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from naviertwin.utils.logger import get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Factory presets — 자주 쓰는 op 별 파라미터 조합
# ---------------------------------------------------------------------------

_FACTORY_PRESETS: dict[str, dict[str, dict[str, Any]]] = {
    "psd_welch": {
        "low_resolution": {"fs": 100.0, "nperseg": 64, "window": "hann"},
        "high_resolution": {"fs": 1000.0, "nperseg": 1024, "window": "hann"},
        "no_window": {"fs": 100.0, "nperseg": 256, "window": "boxcar"},
    },
    "denoise": {
        "light": {"window_length": 5, "polyorder": 2},
        "moderate": {"window_length": 11, "polyorder": 3},
        "aggressive": {"window_length": 31, "polyorder": 4},
    },
    "eof": {
        "few_modes": {"n_modes": 3},
        "default": {"n_modes": 5},
        "many_modes": {"n_modes": 20},
    },
    "auto_report_field": {
        "few_modes": {"n_modes": 3},
        "default": {"n_modes": 5},
        "many_modes": {"n_modes": 15},
    },
    "change_points": {
        "single_change": {"n_changepoints": 1, "method": "binary"},
        "two_changes": {"n_changepoints": 2, "method": "binary"},
        "pelt_auto": {"n_changepoints": 1, "method": "pelt"},
    },
    "phase_average": {
        "1hz": {"period": 1.0, "n_bins": 36},
        "10hz": {"period": 0.1, "n_bins": 36},
        "fine_bins": {"period": 1.0, "n_bins": 90},
    },
    "find_motifs": {
        "short": {"window": 10, "k": 1},
        "medium": {"window": 30, "k": 3},
        "long": {"window": 100, "k": 5},
    },
    "trajectory_clustering": {
        "two_regimes": {"window": 20, "n_clusters": 2},
        "three_regimes": {"window": 20, "n_clusters": 3},
        "fine_window": {"window": 10, "n_clusters": 4},
    },
    "morris_sensitivity": {
        "quick": {"n_trajectories": 5, "n_levels": 4},
        "standard": {"n_trajectories": 10, "n_levels": 4},
        "thorough": {"n_trajectories": 30, "n_levels": 8},
    },
    "permutation_importance": {
        "fast": {"n_repeats": 3},
        "standard": {"n_repeats": 5},
        "robust": {"n_repeats": 20},
    },
    "pod_truncation": {
        "loose_99": {"fraction": 0.99},
        "tight_999": {"fraction": 0.999},
        "very_tight": {"fraction": 0.9999},
    },
    "quantile": {
        "median": {"q": 50.0},
        "p90": {"q": 90.0},
        "p99": {"q": 99.0},
    },
    "morphology_components": {
        "small_blobs": {"threshold": 0.3, "min_size": 4},
        "medium_blobs": {"threshold": 0.5, "min_size": 16},
        "large_blobs": {"threshold": 0.7, "min_size": 64},
    },
    "box_stats": {
        "tukey_15": {"whisker_factor": 1.5},
        "loose_30": {"whisker_factor": 3.0},
        "strict_10": {"whisker_factor": 1.0},
    },
}


def factory_presets(op_name: str) -> dict[str, dict[str, Any]]:
    """op의 사전 정의 프리셋 dict.

    Args:
        op_name: facade op 이름.

    Returns:
        {preset_name: {param: value}} dict. 없으면 빈 dict.
    """
    return dict(_FACTORY_PRESETS.get(op_name, {}))


def list_factory_preset_ops() -> list[str]:
    """프리셋이 정의된 op 이름 목록."""
    return sorted(_FACTORY_PRESETS.keys())


class PresetStore:
    """사용자 정의 프리셋 영속 저장소.

    JSON 파일에 op별 프리셋 dict를 저장. 추가/제거/조회 + factory와 병합.
    """

    def __init__(self, path: str | Path | None = None) -> None:
        self._path = Path(path).expanduser() if path else None
        self._user: dict[str, dict[str, dict[str, Any]]] = {}
        if self._path and self._path.exists():
            self._load()

    def _load(self) -> None:
        try:
            self._user = json.loads(self._path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as e:
            logger.warning("preset 파일 로드 실패: %s", e)
            self._user = {}

    def save(self) -> None:
        if self._path is None:
            return
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._path.write_text(
            json.dumps(self._user, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

    def add(self, op_name: str, preset_name: str, params: dict[str, Any]) -> None:
        """사용자 프리셋 추가/덮어쓰기."""
        if not preset_name:
            raise ValueError("preset_name이 비어있음")
        op_dict = self._user.setdefault(op_name, {})
        op_dict[preset_name] = dict(params)

    def remove(self, op_name: str, preset_name: str) -> bool:
        """프리셋 삭제. 성공 시 True."""
        op_dict = self._user.get(op_name, {})
        if preset_name in op_dict:
            del op_dict[preset_name]
            if not op_dict:
                self._user.pop(op_name, None)
            return True
        return False

    def get(self, op_name: str, preset_name: str) -> dict[str, Any] | None:
        """프리셋 조회 (factory + user 병합. user 우선)."""
        merged = self.merged(op_name)
        return merged.get(preset_name)

    def merged(self, op_name: str) -> dict[str, dict[str, Any]]:
        """factory + user 병합 dict (user 우선)."""
        out = factory_presets(op_name)
        out.update(self._user.get(op_name, {}))
        return out

    def list_presets(self, op_name: str) -> list[str]:
        """op의 모든 프리셋 이름 (factory + user)."""
        return sorted(self.merged(op_name).keys())

    def user_presets(self, op_name: str) -> dict[str, dict[str, Any]]:
        """사용자가 저장한 프리셋만."""
        return dict(self._user.get(op_name, {}))


__all__ = [
    "factory_presets",
    "list_factory_preset_ops",
    "PresetStore",
]
