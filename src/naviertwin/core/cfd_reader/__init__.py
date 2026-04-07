"""CFD 파일 포맷 입출력 모듈.

공개 API:
    - :class:`BaseReader`: CFD 리더 추상 기반 클래스
    - :class:`CFDDataset`: CFD 데이터 컨테이너 데이터클래스
    - :class:`ReaderFactory`: 확장자/디렉토리 구조 기반 자동 감지 팩토리
    - :class:`OpenFOAMReader`: OpenFOAM 케이스 리더
    - :class:`VTKReader`: VTK/VTU/VTP/STL/PLY 파일 리더
    - :class:`FluentReader`: Ansys Fluent .cas/.dat 리더 (ASCII)
    - :class:`CGNSReader`: CGNS .cgns 리더
    - :class:`GmshReader`: Gmsh .msh 리더
"""

from naviertwin.core.cfd_reader.base import BaseReader, CFDDataset
from naviertwin.core.cfd_reader.cgns_reader import CGNSReader  # noqa: F401 — auto-register
from naviertwin.core.cfd_reader.fluent_reader import FluentReader  # noqa: F401 — auto-register
from naviertwin.core.cfd_reader.gmsh_reader import GmshReader  # noqa: F401 — auto-register
from naviertwin.core.cfd_reader.openfoam_reader import OpenFOAMReader
from naviertwin.core.cfd_reader.reader_factory import ReaderFactory
from naviertwin.core.cfd_reader.vtk_reader import VTKReader

# VTKReader / OpenFOAMReader 는 @register 데코레이터가 없으므로 수동 등록
# FluentReader / CGNSReader / GmshReader 는 @ReaderFactory.register 로 자동 등록됨
ReaderFactory.register(VTKReader)
ReaderFactory.register(OpenFOAMReader)

__all__ = [
    "BaseReader",
    "CFDDataset",
    "ReaderFactory",
    "OpenFOAMReader",
    "VTKReader",
    "FluentReader",
    "CGNSReader",
    "GmshReader",
]
