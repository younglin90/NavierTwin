"""PyTorch 동적 양자화 — Linear 모듈을 INT8 로 감싸 메모리/속도 이득.

Examples:
    >>> import torch  # doctest: +SKIP
"""

from __future__ import annotations

from typing import Any


def dynamic_quantize(model: Any):
    """nn.Linear → INT8 동적 양자화 버전."""
    try:
        import torch
        import torch.nn as nn
    except ImportError as exc:
        raise RuntimeError("torch 필요") from exc
    model.eval()
    return torch.quantization.quantize_dynamic(model, {nn.Linear}, dtype=torch.qint8)


def compare_inference(
    model_fp: Any, model_q: Any, sample_input: Any,
    *, atol: float = 1e-2,
) -> dict[str, Any]:
    import numpy as np
    import torch

    model_fp.eval()
    model_q.eval()
    with torch.no_grad():
        y1 = model_fp(sample_input).detach().cpu().numpy()
        y2 = model_q(sample_input).detach().cpu().numpy()
    diff = float(np.max(np.abs(y1 - y2)))
    return {"max_abs_diff": diff, "close": diff < atol}


def model_size_bytes(model: Any) -> int:
    import io

    import torch

    buf = io.BytesIO()
    torch.save(model.state_dict(), buf)
    return len(buf.getvalue())


__all__ = ["dynamic_quantize", "compare_inference", "model_size_bytes"]
