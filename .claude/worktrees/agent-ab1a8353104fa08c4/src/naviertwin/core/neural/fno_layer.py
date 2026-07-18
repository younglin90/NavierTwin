"""FNO — spectral convolution layer (1D/2D).

Li et al. 2021. 단일 layer 경량 구현.

Examples:
    >>> import torch  # doctest: +SKIP
    >>> from naviertwin.core.neural.fno_layer import SpectralConv1d  # doctest: +SKIP
"""

from __future__ import annotations


def _torch():
    try:
        import torch

        return torch
    except ImportError as exc:
        raise RuntimeError("torch 필요") from exc


def SpectralConv1d(in_channels: int, out_channels: int, modes: int):
    """1D Fourier layer."""
    torch = _torch()
    import torch.nn as nn

    class _Conv(nn.Module):
        def __init__(self):
            super().__init__()
            self.in_c = int(in_channels)
            self.out_c = int(out_channels)
            self.modes = int(modes)
            scale = 1.0 / (in_channels * out_channels)
            self.weight = nn.Parameter(
                scale * torch.randn(in_channels, out_channels, modes, dtype=torch.cfloat)
            )

        def forward(self, x):
            # x: (B, C_in, N)
            B, _, N = x.shape
            x_ft = torch.fft.rfft(x, dim=-1)  # (B, C_in, N//2+1)
            out_ft = torch.zeros(B, self.out_c, N // 2 + 1, dtype=torch.cfloat, device=x.device)
            M = min(self.modes, N // 2 + 1)
            out_ft[:, :, :M] = torch.einsum("bim,iom->bom", x_ft[:, :, :M], self.weight[:, :, :M])
            return torch.fft.irfft(out_ft, n=N, dim=-1)

    return _Conv()


def FNOBlock1d(channels: int = 32, modes: int = 8):
    """단일 FNO block: spectral + 1x1 conv + activation."""
    _torch()
    import torch.nn as nn

    class _Block(nn.Module):
        def __init__(self):
            super().__init__()
            self.spec = SpectralConv1d(channels, channels, modes)
            self.w = nn.Conv1d(channels, channels, 1)
            self.act = nn.GELU()

        def forward(self, x):
            return self.act(self.spec(x) + self.w(x))

    return _Block()


__all__ = ["SpectralConv1d", "FNOBlock1d"]
