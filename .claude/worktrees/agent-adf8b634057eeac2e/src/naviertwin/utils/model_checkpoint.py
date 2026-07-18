"""PyTorch 모델 체크포인트 저장/복원.

FNO / DeepONet / PINN 등 nn.Module 기반 모델의 state_dict + 메타데이터를
단일 파일에 저장하여 학습 재개 / 추론 재사용을 지원.

Examples:
    >>> import torch, torch.nn as nn
    >>> from naviertwin.utils.model_checkpoint import save_model, load_model
    >>> m = nn.Linear(3, 2)
    >>> # save_model(m, "m.pt", meta={"epoch": 10, "lr": 1e-3})
    >>> # state = load_model("m.pt"); m.load_state_dict(state["model"])
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from naviertwin.utils.logger import get_logger

logger = get_logger(__name__)


def save_model(
    model: Any,
    path: str | Path,
    *,
    optimizer: Any | None = None,
    meta: dict[str, Any] | None = None,
) -> Path:
    """nn.Module 의 state_dict + 옵션 optimizer/meta 를 torch.save 로 저장.

    Args:
        model: torch.nn.Module (state_dict() 메서드 필요).
        path: 저장 경로 (.pt / .pth 권장).
        optimizer: 선택적 torch.optim.Optimizer (학습 재개용).
        meta: JSON 직렬화 가능한 메타데이터.
    """
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("torch 필요") from exc

    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        "model": model.state_dict(),
        "class_name": type(model).__name__,
        "meta_json": json.dumps(meta or {}, ensure_ascii=False, default=str),
    }
    if optimizer is not None:
        payload["optimizer"] = optimizer.state_dict()
    torch.save(payload, p)
    logger.info("모델 체크포인트 저장: %s (meta=%s)", p, list((meta or {}).keys()))
    return p


def load_model(
    path: str | Path,
    *,
    model: Any | None = None,
    optimizer: Any | None = None,
    map_location: str | None = None,
) -> dict[str, Any]:
    """체크포인트 로드. model 이 주어지면 state_dict 를 load 해둔다.

    Returns:
        {"model": state_dict, "optimizer": state_dict|None, "meta": dict,
         "class_name": str}
    """
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("torch 필요") from exc

    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(p)

    payload = torch.load(p, map_location=map_location, weights_only=False)
    meta_json = payload.get("meta_json", "{}")
    if isinstance(meta_json, bytes):
        meta_json = meta_json.decode()
    meta = json.loads(meta_json) if isinstance(meta_json, str) else dict(meta_json)

    if model is not None:
        model.load_state_dict(payload["model"])
    if optimizer is not None and "optimizer" in payload:
        optimizer.load_state_dict(payload["optimizer"])

    logger.info("모델 체크포인트 로드: %s (class=%s)", p, payload.get("class_name"))
    return {
        "model": payload["model"],
        "optimizer": payload.get("optimizer"),
        "meta": meta,
        "class_name": payload.get("class_name", ""),
    }


def count_parameters(model: Any, trainable_only: bool = True) -> int:
    """nn.Module 의 파라미터 수 합."""
    total = 0
    params = list(model.parameters())
    idx = 0
    if trainable_only:
        while idx < len(params):
            p = params[idx]
            if p.requires_grad:
                total += p.numel()
            idx += 1
        return total
    while idx < len(params):
        total += params[idx].numel()
        idx += 1
    return total


__all__ = ["save_model", "load_model", "count_parameters"]
