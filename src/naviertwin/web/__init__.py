"""NavierTwin 웹 GUI (trame 기반, optional).

데스크톱 PySide6 GUI 의 브라우저 버전. Kitware ``trame`` + ``pyvista.trame`` 으로
동일한 PyVista 렌더 파이프라인을 서버사이드로 스트리밍하며, MVP 워크플로우
(Import → Analyze → Reduce → Twin)를 제공한다.

공개 API:
    - :func:`~naviertwin.web.app.create_web_app`: 웹 앱 생성
    - :func:`~naviertwin.web.app.run_web`: 웹 서버 실행
    - :mod:`naviertwin.web.service`: Qt/GL 비의존 워크플로우 오케스트레이션
    - :mod:`naviertwin.web.render`: Qt 비의존 렌더 메쉬 준비 유틸

trame/trame-vtk/trame-vuetify 가 설치되어 있어야 한다::

    pip install naviertwin[web]
    naviertwin web
"""

from __future__ import annotations

__all__ = ["create_web_app", "run_web"]


def __getattr__(name: str) -> object:
    # trame 미설치 환경에서 ``import naviertwin.web`` 자체는 깨지지 않도록 지연 import.
    if name in __all__:
        from naviertwin.web import app

        return getattr(app, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
