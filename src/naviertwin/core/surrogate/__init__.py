"""대리 모델(Surrogate) 모듈.

공개 API:
    - :class:`BaseSurrogate`: 대리 모델 추상 기반 클래스

구현 예정:
    - RBF 대리 모델 (SMT)
    - Kriging / Co-Kriging (SMT)
    - 베이지안 최적화 연동
"""

from naviertwin.core.surrogate.base import BaseSurrogate

__all__ = ["BaseSurrogate"]
