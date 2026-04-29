"""Fourier, tensorized, wavelet, and spectral neural operators."""

from naviertwin.core.operator_learning.fno.adaptive_fno import AdaptiveFNO1D
from naviertwin.core.operator_learning.fno.fno import FNO1D, FNO2D
from naviertwin.core.operator_learning.fno.lno import LNO1D
from naviertwin.core.operator_learning.fno.spectral_refiner import SpectralRefiner
from naviertwin.core.operator_learning.fno.tfno import TFNO2D
from naviertwin.core.operator_learning.fno.wno import WNO1D

__all__ = [
    "AdaptiveFNO1D",
    "FNO1D",
    "FNO2D",
    "LNO1D",
    "SpectralRefiner",
    "TFNO2D",
    "WNO1D",
]
