"""생성 모델 모듈.

공개 API:
    - :class:`BaseGenerative`: 생성 모델 추상 기반 클래스

하위 모듈:
    - :mod:`diffusion_pde`: PDE 기반 확산 모델
    - :mod:`wavelet_diffusion`: 웨이블릿 확산 신경 연산자
    - :mod:`conditional_gen`: 조건부 생성 모델
"""

from naviertwin.core.generative.base import BaseGenerative

__all__ = ["BaseGenerative"]
