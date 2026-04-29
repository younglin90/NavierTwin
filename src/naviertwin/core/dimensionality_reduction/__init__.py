"""차원 축소 공개 API.

선형(POD, SVD 등)과 비선형(manifold, tensor, autoencoder) reducer를 제공한다.
PyTorch 기반 reducer는 package import 비용을 낮추기 위해 lazy import한다.
"""

from naviertwin.core.dimensionality_reduction.base import BaseReducer
from naviertwin.core.dimensionality_reduction.linear import (
    MRPOD,
    BalancedPOD,
    ConstrainedPOD,
    IncrementalPOD,
    RandomizedPOD,
    SnapshotPOD,
)
from naviertwin.core.dimensionality_reduction.nonlinear import (
    DiffusionMaps,
    TuckerDecomposition,
    isomap,
    lle,
)

__all__ = [
    "Autoencoder",
    "BalancedPOD",
    "BaseReducer",
    "ConstrainedPOD",
    "DiffusionMaps",
    "IncrementalPOD",
    "MRPOD",
    "RandomizedPOD",
    "SnapshotPOD",
    "TuckerDecomposition",
    "VAE",
    "isomap",
    "lle",
]


def __getattr__(name: str) -> object:
    """Lazy import 비선형 reducer — PyTorch 없는 환경에서도 패키지 import 성공."""
    if name == "Autoencoder":
        from naviertwin.core.dimensionality_reduction.nonlinear.autoencoder import Autoencoder

        return Autoencoder
    if name == "VAE":
        from naviertwin.core.dimensionality_reduction.nonlinear.vae import VAE

        return VAE
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
