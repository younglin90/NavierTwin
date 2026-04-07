"""NavierTwin JSON 기반 설정 관리 모듈.

설정은 JSON 파일로 저장되며, :class:`NavierTwinConfig` 데이터클래스로 관리된다.

Examples:
    기본 설정 생성 및 저장::

        from pathlib import Path
        from naviertwin.utils.config import NavierTwinConfig, save_config, load_config

        cfg = NavierTwinConfig()
        save_config(cfg, Path("~/.naviertwin/config.json"))
        cfg2 = load_config(Path("~/.naviertwin/config.json"))
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Literal


@dataclass
class NavierTwinConfig:
    """NavierTwin 전역 설정 데이터클래스.

    Attributes:
        project_dir: 프로젝트 파일(.ntwin)이 저장될 기본 디렉토리.
        log_level: 로깅 레벨. "DEBUG" | "INFO" | "WARNING" | "ERROR" | "CRITICAL".
        gpu_enabled: GPU(CUDA) 사용 여부. False이면 CPU만 사용.
        language: UI 언어. "ko" (한국어) 또는 "en" (영어).
        theme: UI 테마. "dark" 또는 "light".
        max_threads: 병렬 처리에 사용할 최대 스레드 수. 0이면 자동(CPU 코어 수).
        recent_projects: 최근 열었던 프로젝트 경로 목록 (최대 10개).
    """

    project_dir: str = str(Path.home() / "NavierTwinProjects")
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = "INFO"
    gpu_enabled: bool = True
    language: Literal["ko", "en"] = "ko"
    theme: Literal["dark", "light"] = "dark"
    max_threads: int = 0
    recent_projects: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        """초기화 후 유효성 검사를 수행한다.

        Raises:
            ValueError: 유효하지 않은 설정값이 있을 때.
        """
        valid_log_levels = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
        if self.log_level not in valid_log_levels:
            raise ValueError(
                f"log_level은 {valid_log_levels} 중 하나여야 합니다. 받은 값: {self.log_level!r}"
            )
        if self.language not in ("ko", "en"):
            raise ValueError(
                f"language는 'ko' 또는 'en' 이어야 합니다. 받은 값: {self.language!r}"
            )
        if self.theme not in ("dark", "light"):
            raise ValueError(
                f"theme은 'dark' 또는 'light' 이어야 합니다. 받은 값: {self.theme!r}"
            )
        if self.max_threads < 0:
            raise ValueError(
                f"max_threads는 0 이상이어야 합니다. 받은 값: {self.max_threads}"
            )
        # 최근 프로젝트 목록을 최대 10개로 제한
        if len(self.recent_projects) > 10:
            self.recent_projects = self.recent_projects[-10:]


def load_config(path: Path) -> NavierTwinConfig:
    """JSON 파일에서 :class:`NavierTwinConfig`를 로드한다.

    파일이 존재하지 않으면 기본값 설정을 반환한다.

    Args:
        path: 설정 JSON 파일 경로.

    Returns:
        로드된 설정 인스턴스. 파일이 없으면 기본값 인스턴스.

    Raises:
        json.JSONDecodeError: JSON 파싱에 실패한 경우.
        ValueError: 설정값이 유효하지 않은 경우.
    """
    path = Path(path).expanduser().resolve()
    if not path.exists():
        return NavierTwinConfig()

    with path.open(encoding="utf-8") as fp:
        data: dict = json.load(fp)

    # 알 수 없는 키는 무시하고 알려진 키만 사용
    known_fields = {f.name for f in NavierTwinConfig.__dataclass_fields__.values()}  # type: ignore[attr-defined]
    filtered = {k: v for k, v in data.items() if k in known_fields}
    return NavierTwinConfig(**filtered)


def save_config(cfg: NavierTwinConfig, path: Path) -> None:
    """:class:`NavierTwinConfig`를 JSON 파일로 저장한다.

    부모 디렉토리가 없으면 자동으로 생성한다.

    Args:
        cfg: 저장할 설정 인스턴스.
        path: 대상 JSON 파일 경로.

    Raises:
        PermissionError: 파일 시스템 권한이 없는 경우.
    """
    path = Path(path).expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as fp:
        json.dump(asdict(cfg), fp, ensure_ascii=False, indent=2)
