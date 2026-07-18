"""Operator Transformer block — cross-attention 기반 (Cao 2021 Galerkin-Transformer 풍).

Examples:
    >>> import torch
    >>> from naviertwin.core.neural.op_transformer import OpTransformerBlock
    >>> blk = OpTransformerBlock(d_model=16, heads=4)
    >>> x = torch.randn(2, 32, 16)  # (B, N, D)
    >>> y = blk(x)
    >>> y.shape
    torch.Size([2, 32, 16])
"""

from __future__ import annotations

import torch
from torch import nn


class OpTransformerBlock(nn.Module):
    def __init__(self, d_model: int = 64, heads: int = 4, ff_ratio: int = 2) -> None:
        super().__init__()
        self.attn = nn.MultiheadAttention(
            embed_dim=d_model, num_heads=heads, batch_first=True,
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, ff_ratio * d_model),
            nn.GELU(),
            nn.Linear(ff_ratio * d_model, d_model),
        )
        self.norm2 = nn.LayerNorm(d_model)

    def forward(
        self, x: torch.Tensor, *, context: torch.Tensor | None = None,
    ) -> torch.Tensor:
        kv = x if context is None else context
        a, _ = self.attn(self.norm1(x), self.norm1(kv), self.norm1(kv))
        x = x + a
        x = x + self.ff(self.norm2(x))
        return x


__all__ = ["OpTransformerBlock"]
