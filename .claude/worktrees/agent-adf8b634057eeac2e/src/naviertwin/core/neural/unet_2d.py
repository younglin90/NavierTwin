"""2D U-Net — 입력 필드 이미지 → 출력 필드 이미지 매핑.

Examples:
    >>> import torch  # doctest: +SKIP
"""

from __future__ import annotations


def _torch():
    try:
        import torch

        return torch
    except ImportError as exc:
        raise RuntimeError("torch 필요") from exc


def UNet2D(
    in_channels: int = 1, out_channels: int = 1, base: int = 16,
):
    _torch()
    import torch
    import torch.nn as nn

    class _Double(nn.Module):
        def __init__(self, ic, oc):
            super().__init__()
            self.net = nn.Sequential(
                nn.Conv2d(ic, oc, 3, padding=1), nn.GELU(),
                nn.Conv2d(oc, oc, 3, padding=1), nn.GELU(),
            )

        def forward(self, x):
            return self.net(x)

    class _UNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.d1 = _Double(in_channels, base)
            self.d2 = _Double(base, 2 * base)
            self.bot = _Double(2 * base, 4 * base)
            self.up2 = nn.ConvTranspose2d(4 * base, 2 * base, 2, stride=2)
            self.u2 = _Double(4 * base, 2 * base)
            self.up1 = nn.ConvTranspose2d(2 * base, base, 2, stride=2)
            self.u1 = _Double(2 * base, base)
            self.out = nn.Conv2d(base, out_channels, 1)
            self.pool = nn.MaxPool2d(2)

        def forward(self, x):
            x1 = self.d1(x)
            x2 = self.d2(self.pool(x1))
            b = self.bot(self.pool(x2))
            u2 = self.u2(torch.cat([self.up2(b), x2], dim=1))
            u1 = self.u1(torch.cat([self.up1(u2), x1], dim=1))
            return self.out(u1)

    return _UNet()


__all__ = ["UNet2D"]
