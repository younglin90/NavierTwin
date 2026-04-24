"""재현성 보장 — Python/NumPy/Torch 전역 시드 설정.

Examples:
    >>> from naviertwin.utils.seeding import set_global_seed
    >>> set_global_seed(42)
"""

from __future__ import annotations

import os
import random

from naviertwin.utils.logger import get_logger

logger = get_logger(__name__)


def set_global_seed(seed: int = 0, *, deterministic: bool = False) -> None:
    """Python random / NumPy / Torch (CPU+CUDA) 전역 시드."""
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import numpy as np

        np.random.seed(seed)
    except ImportError:
        pass
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        if deterministic:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    except ImportError:
        pass
    logger.info("글로벌 seed=%d (deterministic=%s)", seed, deterministic)


__all__ = ["set_global_seed"]
