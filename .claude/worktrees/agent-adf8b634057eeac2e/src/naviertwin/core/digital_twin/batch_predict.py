"""파이프라인 배치 예측 — 대량 파라미터셋에 대한 field 복원.

NavierTwinPipeline.predict_field 를 대량 입력에 대해 효율적으로 호출.
청크 단위 + 옵션 ThreadPoolExecutor 로 I/O 많은 surrogate 병렬화.

Examples:
    >>> from naviertwin.core.digital_twin.batch_predict import batch_predict_fields
    >>> # fields = batch_predict_fields(pipe, params_matrix, chunk_size=32)
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from typing import Any

import numpy as np
from numpy.typing import NDArray

from naviertwin.utils.logger import get_logger

logger = get_logger(__name__)


def batch_predict_fields(
    pipe: Any,
    params: NDArray[np.float64],
    *,
    chunk_size: int = 64,
    max_workers: int | None = None,
) -> NDArray[np.float64]:
    """(N, n_params) → (n_features, N) 필드 예측.

    Args:
        pipe: NavierTwinPipeline (fit_surrogate 완료).
        params: (N, n_params).
        chunk_size: 한 번에 예측할 파라미터 수.
        max_workers: None → 직렬, >0 → ThreadPoolExecutor.
    """
    if pipe.state.surrogate is None:
        raise RuntimeError("fit_surrogate() 먼저")
    n = params.shape[0]
    chunks = [] if n == 0 else np.split(params, np.arange(chunk_size, n, chunk_size))

    def _one(chunk: NDArray[np.float64]) -> NDArray[np.float64]:
        coeffs = pipe.state.surrogate.predict(chunk)
        return pipe.state.reducer.decode(coeffs)  # (n_features, k)

    if max_workers and max_workers > 1 and len(chunks) > 1:
        with ThreadPoolExecutor(max_workers=max_workers) as ex:
            results = list(ex.map(_one, chunks))
    else:
        results = list(map(_one, chunks))

    out = np.concatenate(results, axis=1)
    logger.info("batch_predict: %d params → %s", n, out.shape)
    return out


__all__ = ["batch_predict_fields"]
