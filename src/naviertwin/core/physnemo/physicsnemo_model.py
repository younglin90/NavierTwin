"""PhysicsNEMO Module 래퍼 — 저장/로드/체크포인트.

PhysicsNEMO 의 Module 는 PyTorch nn.Module 을 상속하며, 메타데이터 + JIT 체크포인트
+ 자동 registry 를 제공한다. 우리의 PINN/FNO 등 자체 모델을 PhysicsNEMO Module
로 감싸면 표준 체크포인트 포맷을 얻을 수 있다.

Examples:
    >>> import torch.nn as nn
    >>> from naviertwin.core.physnemo.physicsnemo_model import wrap_as_physicsnemo_module
    >>> model = nn.Sequential(nn.Linear(4, 8), nn.ReLU(), nn.Linear(8, 2))
    >>> wrapped = wrap_as_physicsnemo_module(model, name="demo")
    >>> hasattr(wrapped, "save")  # PhysicsNEMO 체크포인트 메서드
    True
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from naviertwin.utils.logger import get_logger

logger = get_logger(__name__)


def _require_physicsnemo() -> Any:
    try:
        import physicsnemo

        return physicsnemo
    except ImportError as exc:
        raise RuntimeError(
            "physicsnemo 필요: pip install nvidia-physicsnemo  "
            "(또는 GUI Library 탭에서 PhysicsNeMo 패키지 설치)"
        ) from exc


def wrap_as_physicsnemo_module(
    torch_model: Any,
    name: str = "naviertwin_model",
    description: str = "",
) -> Any:
    """주어진 nn.Module 을 PhysicsNEMO Module 로 감싸 반환.

    체크포인트/메타데이터 저장 기능을 추가한다.
    """
    pnemo = _require_physicsnemo()
    import torch.nn as nn

    if not isinstance(torch_model, nn.Module):
        raise ValueError("torch.nn.Module 이어야 합니다")

    # 최신 PhysicsNEMO 에선 name 은 클래스명 기반 — kwargs 시도 후 fallback
    try:
        meta = pnemo.ModelMetaData()
    except Exception:
        meta = pnemo.ModelMetaData(name=name)

    class _Wrapped(pnemo.Module):
        _meta = meta

        def __init__(self) -> None:
            super().__init__(meta=self._meta)
            self.inner = torch_model

        def forward(self, *args: Any, **kwargs: Any) -> Any:
            return self.inner(*args, **kwargs)

    wrapped = _Wrapped()
    logger.info("PhysicsNEMO Module 래핑: %s", name)
    return wrapped


def save_checkpoint(model: Any, path: str | Path) -> Path:
    """PhysicsNEMO save() 또는 torch.save 폴백."""
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    try:
        model.save(str(out))
    except Exception:
        import torch

        torch.save(model.state_dict(), out)
    logger.info("체크포인트 저장: %s", out)
    return out


def load_checkpoint(
    model: Any, path: str | Path
) -> Any:
    """저장된 가중치 로드."""
    p = Path(path)
    try:
        model.load(str(p))
    except Exception:
        import torch

        state = torch.load(p, map_location="cpu", weights_only=True)
        model.load_state_dict(state)
    logger.info("체크포인트 로드: %s", p)
    return model


def physicsnemo_available() -> bool:
    try:
        import physicsnemo  # noqa: F401

        return True
    except ImportError:
        return False


__all__ = [
    "wrap_as_physicsnemo_module",
    "save_checkpoint",
    "load_checkpoint",
    "physicsnemo_available",
]
