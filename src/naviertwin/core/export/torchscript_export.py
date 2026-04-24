"""PyTorch 모델 → TorchScript 내보내기.

trace / script 두 방식 지원. C++ libtorch 에서 로드 가능.

Examples:
    >>> import torch.nn as nn, tempfile, os
    >>> from naviertwin.core.export.torchscript_export import export_to_torchscript
    >>> model = nn.Linear(4, 2)
    >>> with tempfile.TemporaryDirectory() as d:
    ...     path = os.path.join(d, "m.pt")
    ...     export_to_torchscript(model, sample_input=(__import__("torch").randn(1, 4),),
    ...                            path=path, mode="trace")
    ...     os.path.exists(path)
    True
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from naviertwin.utils.logger import get_logger

logger = get_logger(__name__)


def export_to_torchscript(
    model: Any,
    path: str | Path,
    sample_input: tuple[Any, ...] | None = None,
    mode: str = "trace",
) -> Path:
    """PyTorch 모델을 TorchScript (.pt) 로 저장.

    Args:
        model: torch.nn.Module.
        path: 저장 경로.
        sample_input: trace 모드에서 필수 (예: (tensor,)).
        mode: "trace" 또는 "script".

    Returns:
        저장 경로.
    """
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("torch 가 필요합니다") from exc

    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    model.eval()

    # torch.jit 은 최신 PyTorch 에서 deprecation 경고 — 사용자 명시 경로로만 억제
    import warnings

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        if mode == "trace":
            if sample_input is None:
                raise ValueError("mode='trace' 는 sample_input 필요")
            scripted = torch.jit.trace(model, sample_input)
        elif mode == "script":
            scripted = torch.jit.script(model)
        else:
            raise ValueError(f"mode 는 trace/script: '{mode}'")

    scripted.save(str(out))
    logger.info("TorchScript 저장 완료 (%s): %s", mode, out)
    return out


__all__ = ["export_to_torchscript"]
