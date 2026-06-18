"""NavierTwin GUI public API (PySide6 기반).

GUI 모듈은 core 모듈과 시그널/슬롯으로 통신한다.
core 모듈은 Qt에 의존하지 않는다.

하위 모듈:
    - :mod:`panels`: 주요 탭 패널 (가져오기, 분석, 축소, 모델, 트윈, 내보내기)
    - :mod:`widgets`: 재사용 가능한 위젯 (VTK 뷰어, 플롯 등)
    - :mod:`wizard`: 온보딩 위자드
    - :mod:`styles`: QSS 스타일시트 및 테마
"""

from __future__ import annotations

from importlib import import_module
from typing import Any

_EXPORT_MODULES = {
    "MainWindow": "naviertwin.gui.main_window",
}

__all__ = list(_EXPORT_MODULES)


def __getattr__(name: str) -> Any:
    """Lazily expose Qt entrypoints without eager GUI imports."""
    if name not in _EXPORT_MODULES:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module = import_module(_EXPORT_MODULES[name])
    value = getattr(module, name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    """Return stable public members used by autocomplete and Sphinx."""
    return sorted([*globals(), *__all__])
