"""전이학습 래퍼 — 사전학습 모델을 새 케이스에 파인튜닝.

일반적인 PyTorch 학습 루프를 감싼다. 특정 레이어를 freeze 하고 상위 레이어만
추가 학습하는 옵션 제공.

Examples:
    >>> import torch.nn as nn
    >>> from naviertwin.core.multi_fidelity.transfer_learning import freeze_layers
    >>> model = nn.Sequential(nn.Linear(4, 8), nn.ReLU(), nn.Linear(8, 2))
    >>> freeze_layers(model, n_freeze=1)
    >>> list(map(lambda p: p.requires_grad, model.parameters()))
    [False, False, True, True]
"""

from __future__ import annotations

from typing import Any, Callable

import numpy as np
from numpy.typing import NDArray

from naviertwin.utils.logger import get_logger

logger = get_logger(__name__)


def freeze_layers(model: Any, n_freeze: int) -> None:
    """앞쪽 n_freeze 개 ``nn.Linear`` 레이어 파라미터의 grad 를 false 로."""
    try:
        import torch.nn as nn
    except ImportError as exc:
        raise RuntimeError("torch 필요") from exc

    count = 0
    module_iter = iter(model.modules())
    while True:
        try:
            m = next(module_iter)
        except StopIteration:
            break
        if isinstance(m, nn.Linear):
            if count < n_freeze:
                param_iter = iter(m.parameters())
                while True:
                    try:
                        p = next(param_iter)
                    except StopIteration:
                        break
                    p.requires_grad = False
                count += 1


def finetune(
    model: Any,
    X: NDArray[np.float64],
    Y: NDArray[np.float64],
    loss_fn: Callable[[Any, Any], Any] | None = None,
    lr: float = 1e-4,
    max_epochs: int = 50,
    batch_size: int = 32,
    n_freeze: int = 0,
    device: str = "auto",
) -> list[float]:
    """사전 학습된 모델을 (X, Y) 로 파인튜닝.

    Returns:
        epoch loss 기록.
    """
    import torch
    from torch.utils.data import DataLoader, TensorDataset

    if n_freeze > 0:
        freeze_layers(model, n_freeze)

    dev = torch.device(
        "cuda" if (device == "auto" and torch.cuda.is_available())
        else ("cuda" if device == "cuda" else "cpu")
    )
    model.to(dev)
    params = list(filter(lambda p: p.requires_grad, model.parameters()))
    if not params:
        raise ValueError("학습 가능한 파라미터가 없습니다 (너무 많이 freeze 됨)")
    optim = torch.optim.Adam(params, lr=lr)
    loss_fn = loss_fn or torch.nn.MSELoss()

    X_t = torch.tensor(np.asarray(X, dtype=np.float32))
    Y_t = torch.tensor(np.asarray(Y, dtype=np.float32))
    loader = DataLoader(
        TensorDataset(X_t, Y_t),
        batch_size=min(batch_size, len(X_t)),
        shuffle=True,
    )

    losses: list[float] = []
    epoch_idx = 0
    while epoch_idx < max_epochs:
        epoch = 0.0
        batch_iter = iter(loader)
        while True:
            try:
                xb, yb = next(batch_iter)
            except StopIteration:
                break
            xb = xb.to(dev)
            yb = yb.to(dev)
            optim.zero_grad()
            out = model(xb)
            loss = loss_fn(out, yb)
            loss.backward()
            optim.step()
            epoch += float(loss.item()) * xb.shape[0]
        epoch /= max(len(X_t), 1)
        losses.append(epoch)
        epoch_idx += 1
    logger.info("finetune 완료: epochs=%d, final=%.6g", max_epochs, losses[-1])
    return losses


__all__ = ["freeze_layers", "finetune"]
