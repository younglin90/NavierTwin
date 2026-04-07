"""와류 식별 서브모듈.

공개 API:
    - :func:`compute_q_criterion`: Q-criterion 계산 (pv.compute_derivative)
    - :func:`compute_lambda2`: λ₂ 와류 식별 (numpy eigvalsh)

구현 예정:
    - LCS (Lagrangian Coherent Structures)
"""

from naviertwin.core.flow_analysis.vortex.q_criterion import (
    compute_lambda2,
    compute_q_criterion,
)

__all__ = [
    "compute_q_criterion",
    "compute_lambda2",
]
