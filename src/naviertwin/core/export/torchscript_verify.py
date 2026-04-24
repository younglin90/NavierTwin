"""TorchScript export 결과 검증 — 출력 매칭 확인.

Examples:
    >>> import torch  # doctest: +SKIP
"""

from __future__ import annotations

from pathlib import Path
from typing import Any


def trace_and_save(
    model: Any, sample_input: Any, path: str | Path,
) -> Path:
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("torch 필요") from exc

    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    model.eval()
    traced = torch.jit.trace(model, sample_input)
    traced.save(str(p))
    return p


def verify_script_matches(
    model: Any, sample_input: Any, path: str | Path,
    *, atol: float = 1e-6,
) -> dict[str, Any]:
    """저장된 TorchScript 와 원본 모델 출력 비교."""
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("torch 필요") from exc

    model.eval()
    with torch.no_grad():
        y_ref = model(sample_input)
    loaded = torch.jit.load(str(Path(path)))
    with torch.no_grad():
        y_ts = loaded(sample_input)
    import numpy as np
    diff = float(np.max(np.abs(
        y_ref.detach().cpu().numpy() - y_ts.detach().cpu().numpy()
    )))
    return {"max_abs_diff": diff, "match": diff < atol}


__all__ = ["trace_and_save", "verify_script_matches"]
