"""CFD 파일 포맷 입출력 모듈.

공개 API:
    - :class:`BaseReader`: CFD 리더 추상 기반 클래스
    - :class:`CFDDataset`: CFD 데이터 컨테이너 데이터클래스
"""

from naviertwin.core.cfd_reader.base import BaseReader, CFDDataset

__all__ = [
    "BaseReader",
    "CFDDataset",
]
