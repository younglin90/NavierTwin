"""Attention visualization — Transformer/MultiheadAttention weights 추출.

PyTorch nn.MultiheadAttention 또는 자체 어텐션 모듈에서 attention weight
행렬 (Lq, Lk) 을 추출해 반환. GUI 에서 heatmap 으로 그리기.

Examples:
    >>> import numpy as np
    >>> import torch, torch.nn as nn
    >>> from naviertwin.core.explainability.attention_viz import extract_attention
    >>> mha = nn.MultiheadAttention(embed_dim=8, num_heads=2, batch_first=True)
    >>> x = torch.randn(2, 5, 8)
    >>> out, weights = extract_attention(mha, x)
    >>> weights.shape
    (2, 5, 5)
"""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray


def extract_attention(
    mha: Any, x: Any, mask: Any = None
) -> tuple[Any, NDArray[np.float64]]:
    """nn.MultiheadAttention 의 출력과 average-head attention weight.

    Returns:
        (output, attention_matrix) — weight shape (B, Lq, Lk).
    """
    try:
        import torch  # noqa: F401
    except ImportError as exc:
        raise RuntimeError("torch 필요") from exc

    out, attn = mha(x, x, x, attn_mask=mask, average_attn_weights=True)
    return out, attn.detach().cpu().numpy()


def topk_attention_tokens(
    weights: NDArray[np.float64], k: int = 3
) -> NDArray[np.int64]:
    """각 쿼리 토큰이 가장 크게 가중한 top-k key 인덱스 (B, Lq, k)."""
    if weights.ndim != 3:
        raise ValueError(f"weights (B, Lq, Lk) 3D 필요: {weights.shape}")
    return np.argsort(-weights, axis=-1)[..., :k]


__all__ = ["extract_attention", "topk_attention_tokens"]
