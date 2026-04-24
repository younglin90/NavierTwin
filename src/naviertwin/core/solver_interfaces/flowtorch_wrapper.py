"""flowtorch 래퍼 — 있으면 PyTorch 텐서 파이프라인, 없으면 자체 SVD.

flowtorch 는 GPU 친화적인 POD/DMD 분석을 PyTorch 텐서 위에서 수행한다.
이 래퍼는 간단한 스냅샷 행렬 → 모드/특이값 인터페이스만 노출.

Examples:
    >>> import numpy as np
    >>> from naviertwin.core.solver_interfaces.flowtorch_wrapper import pod_gpu
    >>> X = np.random.default_rng(0).standard_normal((40, 20))
    >>> U, s, V = pod_gpu(X, n_modes=5)
    >>> U.shape
    (40, 5)
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from naviertwin.utils.logger import get_logger

logger = get_logger(__name__)


def pod_gpu(
    snapshots: NDArray[np.float64], n_modes: int
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """PyTorch SVD (GPU 가능) 기반 POD — flowtorch 없이도 동작."""
    try:
        import torch

        x = torch.tensor(snapshots, dtype=torch.float64)
        if torch.cuda.is_available():
            x = x.cuda()
        U, s, Vh = torch.linalg.svd(x, full_matrices=False)
        return (
            U[:, :n_modes].cpu().numpy(),
            s[:n_modes].cpu().numpy(),
            Vh[:n_modes].cpu().numpy().T,
        )
    except ImportError:
        logger.warning("torch 미설치 — numpy SVD 폴백")
        U, s, Vh = np.linalg.svd(snapshots, full_matrices=False)
        return U[:, :n_modes], s[:n_modes], Vh[:n_modes].T


__all__ = ["pod_gpu"]
