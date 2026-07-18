"""Attention-based pooling over set/graph aggregation inputs.

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


def AttentionPool(d_model: int = 64, n_heads: int = 4):
    """learnable query attention → (B, d_model) 벡터."""
    _torch()
    import torch
    import torch.nn as nn

    class _Pool(nn.Module):
        def __init__(self):
            super().__init__()
            self.q = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
            self.attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
            self.norm = nn.LayerNorm(d_model)

        def forward(self, x):
            """x: (B, N, d) → (B, d)."""
            B = x.size(0)
            q = self.q.expand(B, -1, -1)
            out, _ = self.attn(q, x, x, need_weights=False)
            return self.norm(out.squeeze(1))

    return _Pool()


def mean_max_pool(x):
    """(B, N, d) → (B, 2d) concat of mean and max."""
    torch = _torch()
    return torch.cat([x.mean(dim=1), x.max(dim=1).values], dim=-1)


__all__ = ["AttentionPool", "mean_max_pool"]
