"""신경 연산자(Operator Learning) 모듈.

공개 API:
    - :class:`BaseOperator`: 신경 연산자 추상 기반 클래스

하위 모듈:
    - :mod:`fno`: FNO, TFNO, WNO 등 Fourier 기반 신경 연산자
    - :mod:`deeponet`: DeepONet, PI-DeepONet, MIONet 등
    - :mod:`latent_operator`: 잠재 공간 신경 연산자
    - :mod:`koopman`: Koopman 신경 연산자 (KNO, iKNO, FlowDMD)
    - :mod:`kan`: KAN 기반 연산자 (KANO)
    - :mod:`unet`: U-Net 기반 신경 연산자
"""

from naviertwin.core.operator_learning.base import BaseOperator

__all__ = ["BaseOperator"]
