"""Model FLOPs counter — analytic over common layers.

Examples:
    >>> from naviertwin.utils.flops_counter import linear_flops, conv2d_flops
    >>> linear_flops(in_dim=128, out_dim=64, batch=32)
    524288
"""

from __future__ import annotations


def linear_flops(*, in_dim: int, out_dim: int, batch: int = 1) -> int:
    return int(2 * in_dim * out_dim * batch)


def conv2d_flops(
    *, in_ch: int, out_ch: int, kernel: int,
    H_out: int, W_out: int, batch: int = 1,
) -> int:
    return int(2 * in_ch * out_ch * kernel * kernel * H_out * W_out * batch)


def attention_flops(*, batch: int, seq: int, d_model: int, n_heads: int) -> int:
    """Q,K,V proj + scaled dot + softmax (very approximate)."""
    qkv = 3 * batch * seq * d_model * d_model * 2
    attn = batch * n_heads * seq * seq * (d_model // n_heads) * 2
    out = batch * seq * d_model * d_model * 2
    return int(qkv + attn + out)


__all__ = ["attention_flops", "conv2d_flops", "linear_flops"]
