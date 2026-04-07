"""상태 공간 모델(SSM) 모듈.

공개 API:
    - :class:`BaseSSM`: SSM 추상 기반 클래스

하위 모듈:
    - :mod:`mamba_neural_op`: Mamba 기반 신경 연산자 (mamba-ssm)
    - :mod:`deepomamba`: DeepOMamba
"""

from naviertwin.core.state_space.base import BaseSSM

__all__ = ["BaseSSM"]
