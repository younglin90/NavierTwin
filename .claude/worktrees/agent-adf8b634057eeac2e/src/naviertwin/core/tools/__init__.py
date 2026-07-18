"""CFD 전처리 툴 모듈.

공개 API:
    - :func:`generate_channel`, :func:`generate_cylinder`, :func:`generate_airfoil`
      : Gmsh 기반 파라미터 메쉬 생성 (optional ``naviertwin[full]``)
    - :func:`simplify`, :func:`smooth` : PyMeshLab 기반 메쉬 단순화/스무딩
    - :func:`quality_report` : PyVista 기반 셀 품질 지표 (core 만으로 동작)
"""

from naviertwin.core.tools.mesh_generator import (
    generate_airfoil,
    generate_channel,
    generate_cylinder,
)
from naviertwin.core.tools.mesh_processor import (
    quality_report,
    simplify,
    smooth,
)

__all__ = [
    "generate_channel",
    "generate_cylinder",
    "generate_airfoil",
    "simplify",
    "smooth",
    "quality_report",
]
