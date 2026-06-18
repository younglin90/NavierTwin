"""디바이스 유틸 — auto CUDA 감지 + 이동 헬퍼.

Examples:
    >>> from naviertwin.utils.device import get_device
    >>> str(get_device())  # doctest: +ELLIPSIS
    '...'
"""

from __future__ import annotations

from typing import Any


def get_device(prefer: str = "auto"):
    """'auto' | 'cuda' | 'cpu' | 'mps' → torch.device."""
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("torch 필요") from exc
    if prefer == "cpu":
        return torch.device("cpu")
    if prefer == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA unavailable")
        return torch.device("cuda")
    if prefer == "mps":
        if not getattr(torch.backends, "mps", None) or not torch.backends.mps.is_available():
            raise RuntimeError("MPS unavailable")
        return torch.device("mps")
    # auto
    if torch.cuda.is_available():
        return torch.device("cuda")
    if (
        hasattr(torch.backends, "mps")
        and torch.backends.mps.is_available()
    ):
        return torch.device("mps")
    return torch.device("cpu")


def move_to(obj: Any, device: Any) -> Any:
    """Tensor / Module / dict / list / tuple 재귀 이동."""
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("torch 필요") from exc
    if isinstance(obj, torch.Tensor):
        return obj.to(device)
    if isinstance(obj, torch.nn.Module):
        return obj.to(device)
    if isinstance(obj, dict):
        moved: dict[Any, Any] = {}
        items = list(obj.items())
        idx = 0
        while idx < len(items):
            k, v = items[idx]
            moved[k] = move_to(v, device)
            idx += 1
        return moved
    if isinstance(obj, list):
        moved_list: list[Any] = []
        idx = 0
        while idx < len(obj):
            moved_list.append(move_to(obj[idx], device))
            idx += 1
        return moved_list
    if isinstance(obj, tuple):
        moved_tuple: list[Any] = []
        idx = 0
        while idx < len(obj):
            moved_tuple.append(move_to(obj[idx], device))
            idx += 1
        return tuple(moved_tuple)
    return obj


def available_memory_mb() -> dict[str, float]:
    """CUDA 메모리 현황 (MB)."""
    try:
        import torch

        if not torch.cuda.is_available():
            return {"total": 0.0, "free": 0.0, "cuda": False}
        free, total = torch.cuda.mem_get_info()
        return {
            "total": total / 1024 ** 2,
            "free": free / 1024 ** 2,
            "cuda": True,
        }
    except ImportError:
        return {"total": 0.0, "free": 0.0, "cuda": False}


__all__ = ["get_device", "move_to", "available_memory_mb"]
