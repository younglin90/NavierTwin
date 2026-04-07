"""모달 분석 서브모듈.

공개 API:
    - :class:`DMDAnalyzer`: PyDMD 기반 DMD 분석기

구현 예정:
    - SPOD (PySPOD)
    - PGD (Proper Generalized Decomposition)
    - SINDy 방정식 발견 래퍼 (PySINDy)
    - PyKoopman 래퍼
"""

from naviertwin.core.flow_analysis.modal.dmd import DMDAnalyzer

__all__ = ["DMDAnalyzer"]
