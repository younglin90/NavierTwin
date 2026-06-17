"""간단한 JSON 기반 i18n 번역 로더.

Usage:
    >>> from naviertwin.utils.i18n import Translator
    >>> t = Translator(lang="ko")
    >>> t("panel.import")
    'Import (불러오기)'
    >>> t.set_language("en")
    >>> t("panel.import")
    'Import'
"""

from __future__ import annotations

import json
from pathlib import Path

from naviertwin.utils.logger import get_logger

logger = get_logger(__name__)

_DEFAULT_LOCALE_DIR = Path(__file__).resolve().parent.parent / "gui" / "styles" / "i18n"


class Translator:
    """JSON 기반 key → 문자열 번역기."""

    def __init__(self, lang: str = "ko", locale_dir: Path | None = None) -> None:
        self.locale_dir = locale_dir or _DEFAULT_LOCALE_DIR
        self._translations: dict[str, str] = {}
        self.lang = lang
        self.set_language(lang)

    def set_language(self, lang: str) -> None:
        path = self.locale_dir / f"{lang}.json"
        if not path.exists():
            logger.warning("언어 파일 없음: %s — 빈 번역 사용", path)
            self._translations = {}
            self.lang = lang
            return
        with path.open("r", encoding="utf-8") as f:
            self._translations = json.load(f)
        self.lang = lang
        logger.info("언어 설정: %s (%d 키)", lang, len(self._translations))

    def __call__(self, key: str, default: str | None = None) -> str:
        return self._translations.get(key, default if default is not None else key)

    def available_languages(self) -> list[str]:
        if not self.locale_dir.exists():
            return []
        stems: list[str] = []
        paths = list(self.locale_dir.glob("*.json"))
        idx = 0
        while idx < len(paths):
            stems.append(paths[idx].stem)
            idx += 1
        return sorted(stems)


__all__ = ["Translator"]
