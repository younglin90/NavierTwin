"""NavierTwin 프로젝트 재현성 프로파일 — .naviertwin/profile.json 관리.

랜덤 시드, 의존성 버전, 기본 디바이스, 학습 하이퍼파라미터 기본값을 JSON 으로
저장/로드. 같은 프로파일을 다른 머신에서 로드하면 재현 가능.

Examples:
    >>> from naviertwin.utils.profile import Profile
    >>> p = Profile()
    >>> p.set("seed", 42)
    >>> p.set("lr", 1e-3)
    >>> p.get("seed")
    42
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from naviertwin.utils.logger import get_logger

logger = get_logger(__name__)


def _collect_deps() -> dict[str, str]:
    """주요 의존성 버전 수집."""
    versions: dict[str, str] = {}
    libs = [
        "numpy", "scipy", "pyvista", "meshio", "h5py", "torch",
        "sklearn", "smt", "SALib", "botorch", "nlopt", "pydmd",
        "pymor", "PySide6", "foamlib", "pymeshlab",
    ]
    idx = 0
    while idx < len(libs):
        lib = libs[idx]
        try:
            mod = __import__(lib)
            versions[lib] = getattr(mod, "__version__", "unknown")
        except ImportError:
            pass
        idx += 1
    return versions


@dataclass
class Profile:
    """JSON 저장 가능한 설정 프로파일."""

    seed: int = 0
    device: str = "auto"
    lr: float = 1e-3
    default_n_modes: int = 5
    data: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self._meta: dict[str, Any] = {
            "python": sys.version.split()[0],
            "dependencies": _collect_deps(),
        }

    def set(self, key: str, value: Any) -> None:
        if hasattr(self, key) and key != "data":
            setattr(self, key, value)
        else:
            self.data[key] = value

    def get(self, key: str, default: Any = None) -> Any:
        if hasattr(self, key) and key != "data":
            return getattr(self, key)
        return self.data.get(key, default)

    def to_dict(self) -> dict[str, Any]:
        return {
            "seed": self.seed,
            "device": self.device,
            "lr": self.lr,
            "default_n_modes": self.default_n_modes,
            "data": dict(self.data),
            "_meta": self._meta,
        }

    def save(self, path: str | Path) -> Path:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with p.open("w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2, ensure_ascii=False)
        logger.info("Profile 저장: %s", p)
        return p

    @classmethod
    def load(cls, path: str | Path) -> "Profile":
        p = Path(path)
        with p.open("r", encoding="utf-8") as f:
            d = json.load(f)
        inst = cls(
            seed=int(d.get("seed", 0)),
            device=str(d.get("device", "auto")),
            lr=float(d.get("lr", 1e-3)),
            default_n_modes=int(d.get("default_n_modes", 5)),
            data=dict(d.get("data", {})),
        )
        inst._meta = d.get("_meta", inst._meta)
        logger.info("Profile 로드: %s", p)
        return inst

    def apply_seed(self) -> None:
        """numpy + torch + python random seed 일괄 설정."""
        import random

        import numpy as np

        random.seed(self.seed)
        np.random.seed(self.seed)
        try:
            import torch

            torch.manual_seed(self.seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(self.seed)
        except ImportError:
            pass


def default_profile_path() -> Path:
    """기본 위치: ~/.naviertwin/profile.json."""
    return Path.home() / ".naviertwin" / "profile.json"


__all__ = ["Profile", "default_profile_path"]
