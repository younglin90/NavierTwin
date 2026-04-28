"""배치 surrogate 평가 — chunked, 메모리 안전, 진행 콜백.

대용량 입력에 대한 surrogate 예측을 chunked 방식으로 수행. OOM 방지,
optional 진행률 callback, 출력 dtype 제어.

Examples:
    >>> import numpy as np
    >>> def f(X): return np.sum(X ** 2, axis=1)
    >>> X = np.random.default_rng(0).standard_normal((10000, 5))
    >>> from naviertwin.core.surrogate.batch_evaluation import batch_predict
    >>> y = batch_predict(f, X, chunk_size=1000)
    >>> y.shape
    (10000,)
"""

from __future__ import annotations

from typing import Callable

import numpy as np
from numpy.typing import NDArray

from naviertwin.utils.logger import get_logger

logger = get_logger(__name__)


def batch_predict(
    predict_fn: Callable[[NDArray[np.float64]], NDArray[np.float64]],
    X: NDArray[np.float64],
    chunk_size: int = 1024,
    progress_callback: Callable[[int, int], None] | None = None,
    output_dtype: type = np.float64,
) -> NDArray[np.float64]:
    """배치 예측 — chunked.

    Args:
        predict_fn: 함수 — (n, d) → (n,) 또는 (n, k).
        X: (N, d) 또는 (N,) 입력.
        chunk_size: chunk 크기.
        progress_callback: (현재, 총) 매 chunk 호출.
        output_dtype: 출력 dtype.

    Returns:
        예측 출력 ((N,) 또는 (N, k)).

    Raises:
        ValueError: chunk_size ≤ 0 또는 X 빈.
    """
    X = np.asarray(X, dtype=np.float64)
    if chunk_size <= 0:
        raise ValueError(f"chunk_size > 0, got {chunk_size}")
    if X.size == 0:
        raise ValueError("X is empty")

    if X.ndim == 1:
        X = X[:, None]
    N = X.shape[0]

    # 첫 chunk로 출력 형상 결정
    first_chunk = X[: min(chunk_size, N)]
    first_out = np.atleast_1d(predict_fn(first_chunk))
    out_shape = (N,) + first_out.shape[1:]
    out = np.zeros(out_shape, dtype=output_dtype)
    out[: first_out.shape[0]] = first_out.astype(output_dtype)

    if progress_callback is not None:
        progress_callback(first_out.shape[0], N)

    pos = first_out.shape[0]
    while pos < N:
        end = min(pos + chunk_size, N)
        chunk_out = predict_fn(X[pos:end])
        out[pos:end] = np.asarray(chunk_out, dtype=output_dtype)
        pos = end
        if progress_callback is not None:
            progress_callback(pos, N)

    logger.debug("batch_predict 완료: %d 표본 / chunk=%d", N, chunk_size)
    return out


def batch_predict_with_uncertainty(
    predict_mean_fn: Callable[[NDArray[np.float64]], NDArray[np.float64]],
    predict_std_fn: Callable[[NDArray[np.float64]], NDArray[np.float64]],
    X: NDArray[np.float64],
    chunk_size: int = 1024,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """평균 + 표준편차 두 함수를 chunked 평가.

    Args:
        predict_mean_fn: 평균 예측.
        predict_std_fn: 표준편차 예측.
        X: 입력.
        chunk_size: chunk.

    Returns:
        (mean, std).
    """
    mean = batch_predict(predict_mean_fn, X, chunk_size=chunk_size)
    std = batch_predict(predict_std_fn, X, chunk_size=chunk_size)
    return mean, std


def batch_predict_safe(
    predict_fn: Callable[[NDArray[np.float64]], NDArray[np.float64]],
    X: NDArray[np.float64],
    chunk_size: int = 1024,
    fallback_value: float = 0.0,
) -> tuple[NDArray[np.float64], NDArray[np.bool_]]:
    """안전 배치 평가 — 실패한 chunk는 fallback으로 채움 + 마스크 반환.

    Args:
        predict_fn: 예측 함수.
        X: 입력.
        chunk_size: chunk.
        fallback_value: 실패 시 채울 값.

    Returns:
        (predictions, success_mask).
    """
    X = np.asarray(X, dtype=np.float64)
    if X.ndim == 1:
        X = X[:, None]
    N = X.shape[0]

    # 첫 chunk로 출력 형상 결정
    success = np.ones(N, dtype=bool)
    out: NDArray[np.float64] | None = None

    pos = 0
    while pos < N:
        end = min(pos + chunk_size, N)
        try:
            chunk_out = np.asarray(predict_fn(X[pos:end]), dtype=np.float64)
            chunk_out = np.atleast_1d(chunk_out)
            if out is None:
                out_shape = (N,) + chunk_out.shape[1:]
                out = np.full(out_shape, fallback_value, dtype=np.float64)
            out[pos:end] = chunk_out
        except Exception as e:
            logger.warning("chunk [%d:%d] 실패: %s", pos, end, e)
            if out is None:
                # 첫 시도 실패 → fallback 형상은 (N,)
                out = np.full(N, fallback_value, dtype=np.float64)
            success[pos:end] = False
        pos = end

    if out is None:
        out = np.full(N, fallback_value, dtype=np.float64)
    return out, success


def split_into_chunks(
    X: NDArray[np.float64],
    chunk_size: int,
) -> list[tuple[int, int]]:
    """X를 chunk로 분할한 (start, end) 리스트.

    Args:
        X: 입력 배열.
        chunk_size: chunk 크기.

    Returns:
        list of (start_idx, end_idx).

    Raises:
        ValueError: chunk_size ≤ 0.
    """
    if chunk_size <= 0:
        raise ValueError(f"chunk_size > 0, got {chunk_size}")
    X = np.asarray(X)
    N = X.shape[0]
    chunks = []
    pos = 0
    while pos < N:
        end = min(pos + chunk_size, N)
        chunks.append((pos, end))
        pos = end
    return chunks


__all__ = [
    "batch_predict",
    "batch_predict_with_uncertainty",
    "batch_predict_safe",
    "split_into_chunks",
]
