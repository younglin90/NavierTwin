"""유동 분석 모듈.

공개 API:
    - :class:`BaseFlowAnalyzer`: 유동 분석기 추상 기반 클래스

하위 모듈:
    - :mod:`modal`: 모달 분석 (DMD, SPOD, POD, SINDy 등)
    - :mod:`vortex`: 와류 식별 (Q-criterion, λ₂, LCS 등)
    - :mod:`statistics`: 통계 분석 (FFT, PSD, 웨이블릿, 난류 상관 등)
    - :mod:`boundary_layer`: 경계층 분석 (y+, u_tau, δ, θ 등)
    - :mod:`thermofluids`: 열유체 무차원화 (Nu, Re, Pr 등)
"""

from naviertwin.core.flow_analysis.base import BaseFlowAnalyzer

__all__ = ["BaseFlowAnalyzer"]
