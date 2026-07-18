"""비선형 차원 축소 및 manifold ROM 공개 API.

Torch 계열 모델은 고객이 해당 심볼을 요청할 때만 import한다.
"""

from naviertwin.core.dimensionality_reduction.nonlinear.adaptive_enrich import (
    enrich_basis,
)
from naviertwin.core.dimensionality_reduction.nonlinear.closure_rom import (
    apply_closure,
    fit_closure,
)
from naviertwin.core.dimensionality_reduction.nonlinear.diffusion_maps import (
    DiffusionMaps,
)
from naviertwin.core.dimensionality_reduction.nonlinear.isomap import isomap
from naviertwin.core.dimensionality_reduction.nonlinear.latent_linearize import (
    fit_latent_A,
    predict_latent,
)
from naviertwin.core.dimensionality_reduction.nonlinear.lift_learn import (
    fit_lifted_linear,
    lift_polynomial,
)
from naviertwin.core.dimensionality_reduction.nonlinear.lle import lle
from naviertwin.core.dimensionality_reduction.nonlinear.manifold_tangent import (
    tangent_space,
)
from naviertwin.core.dimensionality_reduction.nonlinear.opinf import opinf_fit
from naviertwin.core.dimensionality_reduction.nonlinear.reduced_barycentric import (
    barycentric_weights,
)
from naviertwin.core.dimensionality_reduction.nonlinear.tucker_decomp import (
    TuckerDecomposition,
)

__all__ = [
    "Autoencoder",
    "CNNAE",
    "DiffusionMaps",
    "GNNAE",
    "LatentODE",
    "TuckerDecomposition",
    "VAE",
    "apply_closure",
    "barycentric_weights",
    "enrich_basis",
    "fit_closure",
    "fit_latent_A",
    "fit_lifted_linear",
    "isomap",
    "lift_polynomial",
    "lle",
    "opinf_fit",
    "predict_latent",
    "tangent_space",
]


def __getattr__(name: str) -> object:
    """Lazy import torch-backed nonlinear reducers."""
    if name == "Autoencoder":
        from naviertwin.core.dimensionality_reduction.nonlinear.autoencoder import Autoencoder

        return Autoencoder
    if name == "CNNAE":
        from naviertwin.core.dimensionality_reduction.nonlinear.cnn_ae import CNNAE

        return CNNAE
    if name == "GNNAE":
        from naviertwin.core.dimensionality_reduction.nonlinear.gnn_ae import GNNAE

        return GNNAE
    if name == "LatentODE":
        from naviertwin.core.dimensionality_reduction.nonlinear.latent_ode import LatentODE

        return LatentODE
    if name == "VAE":
        from naviertwin.core.dimensionality_reduction.nonlinear.vae import VAE

        return VAE
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
