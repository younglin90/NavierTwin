"""NavierTwin 공통 로거 팩토리 모듈.

rotating file handler와 stream handler를 모두 등록한 로거를 제공한다.
로그 파일은 ``~/.naviertwin/logs/naviertwin.log`` 에 저장되며
최대 5MB × 3개로 교체된다.

Examples:
    일반적인 사용법::

        from naviertwin.utils.logger import get_logger

        logger = get_logger(__name__)
        logger.info("작업 시작")
        logger.warning("경고 메시지: %s", detail)
"""

from __future__ import annotations

import logging
import logging.handlers
from pathlib import Path

_LOG_DIR = Path.home() / ".naviertwin" / "logs"
_LOG_FILE = _LOG_DIR / "naviertwin.log"
_MAX_BYTES = 5 * 1024 * 1024  # 5 MB
_BACKUP_COUNT = 3
_FMT = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
_DATE_FMT = "%Y-%m-%d %H:%M:%S"

# 루트 로거 중복 핸들러 방지용 플래그
_root_configured: bool = False


def _configure_root_logger(level: int = logging.DEBUG) -> None:
    """NavierTwin 전용 루트 로거를 한 번만 설정한다.

    Args:
        level: 루트 로거에 적용할 기본 로그 레벨.
    """
    global _root_configured  # noqa: PLW0603
    if _root_configured:
        return

    root = logging.getLogger("naviertwin")
    root.setLevel(level)

    formatter = logging.Formatter(fmt=_FMT, datefmt=_DATE_FMT)

    # --- 스트림 핸들러 (콘솔 출력) ---
    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(logging.INFO)
    stream_handler.setFormatter(formatter)
    root.addHandler(stream_handler)

    # --- Rotating File 핸들러 ---
    try:
        _LOG_DIR.mkdir(parents=True, exist_ok=True)
        file_handler = logging.handlers.RotatingFileHandler(
            filename=_LOG_FILE,
            maxBytes=_MAX_BYTES,
            backupCount=_BACKUP_COUNT,
            encoding="utf-8",
        )
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(formatter)
        root.addHandler(file_handler)
    except OSError as exc:
        root.warning("로그 파일 핸들러를 열 수 없습니다: %s", exc)

    # 상위 로거(root)로 전파 방지 (중복 출력 억제)
    root.propagate = False
    _root_configured = True


def get_logger(name: str, level: str = "DEBUG") -> logging.Logger:
    """지정된 이름의 로거를 반환한다.

    최초 호출 시 NavierTwin 루트 로거("naviertwin")를 설정한다.
    ``name`` 이 "naviertwin"으로 시작하지 않으면 자동으로 접두사를 붙인다.

    Args:
        name: 로거 이름. 보통 ``__name__`` 을 전달한다.
        level: 이 로거에 적용할 레벨 문자열 (기본값: "DEBUG").
            부모 로거의 레벨보다 낮으면 실질적으로 무시된다.

    Returns:
        설정된 :class:`logging.Logger` 인스턴스.

    Examples:
        >>> logger = get_logger(__name__)
        >>> logger.info("모듈 초기화 완료")
    """
    _configure_root_logger()

    # naviertwin 네임스페이스 하위에 항상 배치
    if not name.startswith("naviertwin"):
        full_name = f"naviertwin.{name}"
    else:
        full_name = name

    logger = logging.getLogger(full_name)
    numeric_level = getattr(logging, level.upper(), logging.DEBUG)
    logger.setLevel(numeric_level)
    return logger
