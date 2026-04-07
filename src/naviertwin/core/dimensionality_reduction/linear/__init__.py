"""선형 차원 축소 기법 모듈.

공개 API:
    - :class:`SnapshotPOD`: SVD 기반 스냅샷 POD
    - :class:`RandomizedPOD`: Randomized SVD 기반 고속 POD

구현 예정:
    - Certified Reduced Basis
    - CPOD
"""

from naviertwin.core.dimensionality_reduction.linear.pod import SnapshotPOD
from naviertwin.core.dimensionality_reduction.linear.randomized_svd import RandomizedPOD

__all__ = ["SnapshotPOD", "RandomizedPOD"]
