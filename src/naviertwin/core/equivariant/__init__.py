"""등변 신경망(Equivariant Neural Network) 모듈.

공개 API:
    - :class:`BaseEquivariant`: 등변 모델 추상 기반 클래스

하위 모듈:
    - :mod:`group_equiv_fno`: 그룹 등변 FNO (e3nn)
    - :mod:`physics_embedded`: 물리 내장 GNN (escnn, Lie algebra NO)
"""

from naviertwin.core.equivariant.base import BaseEquivariant

__all__ = ["BaseEquivariant"]
