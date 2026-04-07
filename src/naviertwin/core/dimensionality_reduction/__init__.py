"""차원 축소 모듈.

선형(POD, SVD 등)과 비선형(오토인코더, VAE 등) 차원 축소 기법을 제공한다.

공개 API:
    - :class:`BaseReducer`: 차원 축소기 추상 기반 클래스
"""

from naviertwin.core.dimensionality_reduction.base import BaseReducer

__all__ = ["BaseReducer"]
