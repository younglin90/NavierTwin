"""경량 Transformer encoder block — Pre-LN + MHA + FFN.

시계열 surrogate / sequence-to-sequence ROM 용.

Examples:
    >>> import torch  # doctest: +SKIP
    >>> from naviertwin.core.neural.transformer_block import TransformerEncoderBlock
    >>> blk = TransformerEncoderBlock(d_model=32, n_heads=4)  # doctest: +SKIP
"""

from __future__ import annotations


def _torch():
    try:
        import torch

        return torch
    except ImportError as exc:
        raise RuntimeError("torch 필요") from exc


def TransformerEncoderBlock(
    d_model: int = 64, n_heads: int = 4, d_ff: int | None = None,
    dropout: float = 0.0, activation: str = "gelu",
):
    """Pre-norm encoder block: x + MHA(LN(x)) → x + FFN(LN(x))."""
    _torch()
    import torch.nn as nn

    if d_ff is None:
        d_ff = 4 * d_model

    class _Block(nn.Module):
        def __init__(self):
            super().__init__()
            self.ln1 = nn.LayerNorm(d_model)
            self.mha = nn.MultiheadAttention(
                d_model, n_heads, dropout=dropout, batch_first=True,
            )
            self.ln2 = nn.LayerNorm(d_model)
            act = nn.GELU() if activation == "gelu" else nn.ReLU()
            self.ffn = nn.Sequential(
                nn.Linear(d_model, d_ff), act, nn.Dropout(dropout),
                nn.Linear(d_ff, d_model), nn.Dropout(dropout),
            )

        def forward(self, x, mask=None):
            y = self.ln1(x)
            a, _ = self.mha(y, y, y, need_weights=False, attn_mask=mask)
            x = x + a
            x = x + self.ffn(self.ln2(x))
            return x

    return _Block()


def positional_encoding(seq_len: int, d_model: int):
    """sinusoidal positional encoding (seq_len, d_model) tensor."""
    torch = _torch()
    pos = torch.arange(seq_len, dtype=torch.float32).unsqueeze(1)
    div = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float32) *
                    (-torch.log(torch.tensor(10000.0)) / d_model))
    pe = torch.zeros(seq_len, d_model)
    pe[:, 0::2] = torch.sin(pos * div)
    pe[:, 1::2] = torch.cos(pos * div)
    return pe


__all__ = ["TransformerEncoderBlock", "positional_encoding"]
