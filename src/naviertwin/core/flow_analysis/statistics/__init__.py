"""통계 분석 서브모듈.

공개 API:
    - :func:`compute_fft`: 1D 시계열 FFT
    - :func:`compute_psd`: Welch PSD
    - :func:`find_dominant_frequencies`: 지배 주파수 피크 탐색
    - :func:`compute_field_fft`: 공간 필드 스냅샷 FFT

구현 예정:
    - 웨이블릿 분석
    - 난류 두 점 공간 상관 함수
"""

from naviertwin.core.flow_analysis.statistics.fft_psd import (
    compute_fft,
    compute_field_fft,
    compute_psd,
    find_dominant_frequencies,
)

__all__ = [
    "compute_fft",
    "compute_psd",
    "find_dominant_frequencies",
    "compute_field_fft",
]
